"""Core orchestration logic: plan subtasks and route to agents."""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from .agent_registry import AgentRegistry
from .cost import CostEstimate, estimate_cost
from .llm_client import LLMUsage, UsageAccumulator, call_llm_json
from . import prompts
from .schemas import OrchestratorOutput

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Simple token-bucket rate limiter (requests per minute)."""

    def __init__(self, rpm: int) -> None:
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


class Orchestrator:
    """Plans subtasks from a user task and assigns each to the best agent."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        config_path = config_path or Path(__file__).parent / "config.yaml"
        self._config = self._load_config(Path(config_path))

        self._model: str = self._config["llm"]["default_model"]
        self._temperature: float = self._config["llm"].get("temperature", 0.2)
        self._max_tokens: int = self._config["llm"].get("max_tokens", 4096)
        self._api_base: str | None = self._config["llm"].get("api_base")
        self._max_retries: int = self._config.get("retry", {}).get("max_retries", 3)
        self._fallback_model: str | None = self._config.get("retry", {}).get("fallback_model")
        # Detect Claude family to enable Anthropic prompt caching (cache_control
        # on the static system + registry blocks). Matches yunwu/claude-*,
        # anthropic/claude-*, openrouter/anthropic/claude-*, etc.
        self._claude_caching: bool = "claude" in (self._model or "").lower()

        concurrency_cfg = self._config.get("concurrency", {})
        self._max_workers: int = concurrency_cfg.get("max_workers", 5)
        self._rpm: int = concurrency_cfg.get("requests_per_minute", 60)

        agents_dir = self._resolve_path(self._config.get("agents_dir", "../agents"))
        self._registry = AgentRegistry()
        self._registry.load_dir(agents_dir)

        self._usage = UsageAccumulator()

        logger.info(
            "Orchestrator ready – %d agents, model=%s, workers=%d, rpm=%d",
            len(self._registry), self._model, self._max_workers, self._rpm,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def usage(self) -> UsageAccumulator:
        return self._usage

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def get_cost_estimate(self) -> CostEstimate:
        """Compute cost estimate from accumulated usage."""
        return estimate_cost(
            self._model,
            self._usage.total_prompt_tokens,
            self._usage.total_completion_tokens,
        )

    def run(self, task: str) -> OrchestratorOutput:
        """Orchestrate: decompose *task* into subtasks and assign agents."""
        json_schema = json.dumps(OrchestratorOutput.model_json_schema(), indent=2)
        system_msg = prompts.format_system_prompt(json_schema)
        registry_text = self._registry.to_prompt_text()

        extra_headers: dict[str, str] | None = None
        if self._claude_caching:
            # Anthropic prompt caching: keep the static system + registry blocks
            # behind cache_control breakpoints so only the per-task delta is
            # billed at the fresh-input rate after the first call.
            task_block = (
                f"## User Task\n\n{task}\n\n"
                "Decompose the above task into subtasks and assign each to the "
                "best agent. Return the result as JSON."
            )
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_msg, "cache_control": {"type": "ephemeral"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"## Agent Registry\n\n{registry_text}",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": task_block},
                    ],
                },
            ]
            extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
        else:
            user_msg = prompts.format_user_prompt(task, registry_text)
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

        raw, token_usage = self._call_with_retry(messages, extra_headers=extra_headers)
        self._usage.add(token_usage)

        result = OrchestratorOutput.model_validate(raw)
        result.original_task = task
        return result

    def load_dataset_tasks(self, dataset_name: str) -> list[dict]:
        """Load tasks from a dataset's task.json without running them."""
        datasets_dir = self._resolve_path(self._config.get("datasets_dir", "../Datasets"))
        task_file = datasets_dir / dataset_name / "task.json"
        if not task_file.exists():
            raise FileNotFoundError(f"task.json not found at {task_file}")
        return json.loads(task_file.read_text(encoding="utf-8"))

    def run_dataset(
        self,
        dataset_name: str,
        output_path: Path | None = None,
        max_workers: int | None = None,
    ) -> list[dict]:
        """Run the orchestrator over every task in a dataset's task.json.

        Supports incremental saving, resumption, and parallel execution.
        """
        tasks = self.load_dataset_tasks(dataset_name)
        total = len(tasks)
        workers = max_workers or self._max_workers

        completed: dict[int, dict] = {}
        save_lock = threading.Lock()

        if output_path and output_path.exists():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                for entry in existing:
                    idx = entry.get("task_index")
                    if idx is not None:
                        completed[idx] = entry
                if completed:
                    logger.info(
                        "Resuming: found %d/%d completed tasks in %s",
                        len(completed), total, output_path,
                    )
            except (json.JSONDecodeError, KeyError):
                logger.warning("Could not parse existing results, starting fresh")

        pending = [
            (i, entry) for i, entry in enumerate(tasks) if i not in completed
        ]
        if not pending:
            logger.info("All %d tasks already completed", total)
            return [completed[i] for i in range(total) if i in completed]

        rate_limiter = _RateLimiter(self._rpm)
        done_count = len(completed)

        if workers <= 1:
            for i, entry in pending:
                self._process_one(
                    i, entry, total, completed, save_lock, output_path, rate_limiter,
                )
        else:
            logger.info("Parallel mode: %d workers, RPM limit=%d", workers, self._rpm)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self._process_one,
                        i, entry, total, completed, save_lock, output_path, rate_limiter,
                    ): i
                    for i, entry in pending
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error("Task index %d failed: %s", idx, exc)

        ordered = [completed[i] for i in range(total) if i in completed]
        return ordered

    def estimate_prompt_tokens(self, sample_task: str = "sample task") -> int:
        """Estimate prompt token count for a single task (without calling LLM)."""
        json_schema = json.dumps(OrchestratorOutput.model_json_schema(), indent=2)
        system_msg = prompts.format_system_prompt(json_schema)
        user_msg = prompts.format_user_prompt(sample_task, self._registry.to_prompt_text())
        full_text = system_msg + user_msg
        return len(full_text) // 3

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_one(
        self,
        i: int,
        entry: dict,
        total: int,
        completed: dict[int, dict],
        save_lock: threading.Lock,
        output_path: Path | None,
        rate_limiter: _RateLimiter,
    ) -> None:
        """Process a single task: rate-limit, call LLM, save incrementally."""
        desc = entry.get("task_description") or entry.get("task_inst", "")
        task_id = entry.get("id", i + 1)
        logger.info("[%d/%d] Processing: %s", i + 1, total, desc[:80])

        rate_limiter.acquire()
        result = self.run(desc)

        result_dict = result.model_dump()
        result_dict["task_index"] = i
        result_dict["task_id"] = task_id

        with save_lock:
            completed[i] = result_dict
            if output_path:
                self._atomic_save(output_path, completed, total)
            done_now = len(completed)

        cost = self.get_cost_estimate()
        logger.info(
            "[%d/%d done] Cumulative: %d calls, %d tokens, ¥%.4f",
            done_now, total,
            self._usage.num_calls,
            self._usage.total_tokens,
            cost.total_cost,
        )

    @staticmethod
    def _atomic_save(path: Path, completed: dict[int, dict], total: int) -> None:
        """Write results to file atomically (write tmp then rename)."""
        ordered = [completed[i] for i in range(total) if i in completed]
        data = json.dumps(ordered, ensure_ascii=False, indent=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp", prefix=path.stem,
        )
        try:
            with open(tmp_fd, "w", encoding="utf-8") as f:
                f.write(data)
            Path(tmp_path).replace(path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        logger.debug("Saved %d/%d results to %s", len(completed), total, path)

    def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[dict, LLMUsage]:
        last_exc: Exception | None = None
        models = [self._model]
        if self._fallback_model and self._fallback_model != self._model:
            models.append(self._fallback_model)

        for model in models:
            for attempt in range(1, self._max_retries + 1):
                try:
                    logger.info("Attempt %d with model=%s", attempt, model)
                    result, usage = call_llm_json(
                        model,
                        messages,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        api_base=self._api_base,
                        extra_headers=extra_headers,
                    )
                    return result, usage
                except Exception as exc:
                    last_exc = exc
                    logger.warning("Attempt %d failed (%s): %s", attempt, model, exc)

        raise RuntimeError(f"All retries exhausted. Last error: {last_exc}") from last_exc

    def _resolve_path(self, rel: str) -> Path:
        """Resolve a path relative to the config file's directory."""
        return (Path(__file__).parent / rel).resolve()

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return yaml.safe_load(path.read_text(encoding="utf-8"))
