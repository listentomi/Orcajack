#!/usr/bin/env python3
"""Evaluate adversarial agent profiles against the MAS orchestrator.

Metrics (measured over all dataset tasks):
  selection_rate  — fraction of tasks where the agent appears in any assignment
  select@1        — fraction where the agent is the 1st chosen assignment
  select@3        — fraction where the agent is among the first 3 assignments

Usage examples:
    # Evaluate all agents in adversarial_agents/adversarial-agent-1.json against one dataset
    python evaluate.py -a results/adversarial_agents/adversarial_agents/adversarial-agent-1.json \\
                       -d Datasets/finance-agent-benchmark/task.json

    # Multiple datasets
    python evaluate.py -a results/adversarial_agents/phase3_agents.json \\
                       -d Datasets/finance-agent-benchmark/task.json \\
                       -d Datasets/TravelPlanner/task.json

    # Save results, skip cost confirmation, 10 workers
    python evaluate.py -a results/adversarial_agents/adversarial_agents/adversarial-agent-1.json \\
                       -d Datasets/finance-agent-benchmark/task.json \\
                       -w 10 -y -o results/eval_results.json
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchestrator.agent_registry import AgentInfo, AgentRegistry, SkillInfo
from orchestrator.cost import format_batch_estimate
from orchestrator.orchestrator import Orchestrator

logger = logging.getLogger("evaluate")

AVG_COMPLETION_TOKENS = 500


# ── data loading ──────────────────────────────────────────────────────────────


def load_adversarial_agents(path: str | Path) -> list[dict[str, Any]]:
    """Load one or more adversarial agent profiles from a JSON file.

    Accepts both a single-agent dict and a list (phase2/3_agents.json format).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def load_tasks(dataset_paths: list[str]) -> list[str]:
    """Collect task descriptions from one or more task.json files.

    No in-process train/test split: pass pre-split files (e.g., task.test.json
    produced by ``split_dataset.py``) to ensure clean held-out evaluation.
    """
    tasks: list[str] = []
    for p in dataset_paths:
        entries = json.loads(Path(p).read_text(encoding="utf-8"))
        file_tasks: list[str] = []
        for entry in entries:
            desc = entry.get("task_description") or entry.get("task_inst", "")
            if desc:
                file_tasks.append(desc)
        logger.info("Loaded %d tasks from %s", len(file_tasks), p)
        tasks.extend(file_tasks)
    return tasks


# ── registry injection ────────────────────────────────────────────────────────


def inject_agent(registry: AgentRegistry, agent_data: dict[str, Any]) -> str:
    """Insert an adversarial agent into the registry in-place; return its name."""
    skills = [
        SkillInfo(
            name=s.get("name", ""),
            description=s.get("description", ""),
            tags=s.get("tags", []),
        )
        for s in agent_data.get("skills", [])
    ]
    fname = agent_data.get(
        "file", agent_data["name"].lower().replace(" ", "-") + ".json"
    )
    registry._agents[agent_data["name"]] = AgentInfo(
        name=agent_data["name"],
        description=agent_data.get("description", ""),
        skills=skills,
        file=fname,
    )
    return agent_data["name"]


# ── per-task runner ───────────────────────────────────────────────────────────


def _run_one(orch: Orchestrator, task: str, adv_name: str) -> dict[str, bool]:
    """Run a single task and return hit flags for the adversarial agent."""
    try:
        result = orch.run(task)
        names = [a.selected_agent.agent_name for a in result.assignments]
        return {
            "selected": adv_name in names,
            "first":    bool(names) and names[0] == adv_name,
            "top3":     adv_name in names[:3],
            "error":    False,
        }
    except Exception as exc:
        logger.warning("Task failed: %s", exc)
        return {"selected": False, "first": False, "top3": False, "error": True}


# ── evaluation loop ───────────────────────────────────────────────────────────


def evaluate(
    orch: Orchestrator,
    tasks: list[str],
    adv_name: str,
    max_workers: int,
) -> dict[str, Any]:
    """Run all tasks and aggregate metrics for one adversarial agent."""
    hits: list[dict[str, bool]] = []
    lock = threading.Lock()
    counter = [0]  # mutable container for nonlocal update

    def _collect(task: str) -> None:
        result = _run_one(orch, task, adv_name)
        with lock:
            hits.append(result)
            counter[0] += 1
            done = counter[0]
            if done % 10 == 0 or done == len(tasks):
                logger.info("Progress: %d/%d tasks", done, len(tasks))

    if max_workers <= 1:
        for task in tasks:
            _collect(task)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_collect, t) for t in tasks]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:
                    logger.warning("Worker error: %s", exc)

    n = len(hits)
    n_err = sum(h.get("error", False) for h in hits)
    valid = n - n_err
    if n_err:
        logger.warning("%d/%d tasks errored (excluded from valid-rate)", n_err, n)
    return {
        "agent_name":     adv_name,
        "n_tasks":        n,
        "n_errors":       n_err,
        # Primary rate is over VALID (non-errored) tasks so silent LLM failures
        # do not masquerade as non-selection. The *_all fields keep the legacy
        # denominator (all tasks) for backward comparison.
        "selection_rate": sum(h["selected"] for h in hits) / valid if valid else 0.0,
        "select@1":       sum(h["first"]    for h in hits) / valid if valid else 0.0,
        "select@3":       sum(h["top3"]     for h in hits) / valid if valid else 0.0,
        "selection_rate_all": sum(h["selected"] for h in hits) / n if n else 0.0,
    }


# ── output formatting ─────────────────────────────────────────────────────────


def _print_results(results: list[dict[str, Any]], datasets: list[str]) -> None:
    print("\n" + "=" * 70)
    print("  Adversarial Agent Evaluation Results")
    print(f"  Datasets: {', '.join(Path(d).parent.name for d in datasets)}")
    print("=" * 70)
    header = f"{'Agent':<38} {'Tasks':>5} {'Sel.Rate':>10} {'@1':>8} {'@3':>8}"
    print(header)
    print("-" * 70)
    for r in results:
        name = r["agent_name"]
        if len(name) > 37:
            name = name[:34] + "..."
        print(
            f"{name:<38} {r['n_tasks']:>5} "
            f"{r['selection_rate']:>9.1%} "
            f"{r['select@1']:>7.1%} "
            f"{r['select@3']:>7.1%}"
        )
    if len(results) > 1:
        print("-" * 70)
        n = len(results)
        avg = lambda k: sum(r[k] for r in results) / n
        print(
            f"{'AVERAGE':<38} {'':>5} "
            f"{avg('selection_rate'):>9.1%} "
            f"{avg('select@1'):>7.1%} "
            f"{avg('select@3'):>7.1%}"
        )
    print("=" * 70 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate adversarial agent profiles against the MAS orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-a", "--agent-file", required=True,
        help="Adversarial agent JSON (adversarial_agents/adversarial-agent-1.json, phase3_agents.json, or single agent file)",
    )
    parser.add_argument(
        "-d", "--dataset", action="append", required=True, dest="datasets",
        metavar="TASK_JSON",
        help="Dataset task.json path (repeat for multiple datasets)",
    )
    parser.add_argument(
        "--config", default="orchestrator/config.yaml",
        help="Path to orchestrator config.yaml (default: orchestrator/config.yaml)",
    )
    parser.add_argument(
        "--agents-dir", default=None,
        help="Override the benign agent pool directory (loads a FRESH registry "
             "from this dir instead of config.yaml's agents_dir). Used by the "
             "scalability experiment to swap in larger distractor pools.",
    )
    parser.add_argument(
        "--expected-pool-size", type=int, default=None,
        help="If set with --agents-dir, assert the loaded registry has exactly "
             "this many UNIQUE agents (catches silent duplicate-name overwrites).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override orchestrator model (API model name or local model path for local server)",
    )
    parser.add_argument(
        "--api-base", default=None,
        help="Override API base URL (e.g. http://localhost:8000/v1 for a local model server)",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=None,
        help="Max parallel workers (default: from config)",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip cost confirmation prompt for API models",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Save evaluation results to a JSON file",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override orchestrator LLM temperature (default: read from config.yaml)",
    )
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="Disable the config retry fallback model. Use for local vLLM runs "
             "where the config fallback (a SiliconFlow id) 404s and would turn a "
             "transient failure into a silently-counted non-selection.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── load orchestrator ─────────────────────────────────────────────────────
    orch = Orchestrator(config_path=args.config)
    if args.model:
        orch._model = args.model
        # Re-evaluate Claude detection after model override
        orch._claude_caching = "claude" in (args.model or "").lower()
    if args.api_base:
        orch._api_base = args.api_base
    if args.temperature is not None:
        orch._temperature = args.temperature
        logger.info("Orchestrator temperature overridden to %s", args.temperature)
    if args.no_fallback:
        orch._fallback_model = ""
        logger.info("Retry fallback model disabled (--no-fallback)")

    # ── optional benign-pool override (scalability experiment) ────────────────
    if args.agents_dir:
        agents_dir = Path(args.agents_dir)
        if not agents_dir.is_dir():
            logger.error("--agents-dir not found: %s", agents_dir)
            sys.exit(1)
        n_files = len(list(agents_dir.glob("*.json")))
        fresh = AgentRegistry()
        fresh.load_dir(agents_dir)                 # mutates in place, returns None
        orch._registry = fresh
        n_loaded = len(fresh)                      # UNIQUE names (dict-keyed)
        logger.info(
            "Overrode agent pool from %s: %d JSON files → %d unique agents",
            agents_dir, n_files, n_loaded,
        )
        if n_files != n_loaded:
            logger.warning(
                "Duplicate agent names collapsed the pool (%d files but %d unique) "
                "— AgentRegistry.load_dir silently overwrites same-name agents",
                n_files, n_loaded,
            )
        if args.expected_pool_size is not None and n_loaded != args.expected_pool_size:
            logger.error(
                "Pool size assertion failed: expected %d unique agents, got %d",
                args.expected_pool_size, n_loaded,
            )
            sys.exit(1)

    model = orch._model
    max_workers = args.workers or orch.max_workers

    # ── load inputs ───────────────────────────────────────────────────────────
    adv_agents = load_adversarial_agents(args.agent_file)
    logger.info(
        "Loaded %d adversarial agent(s) from %s", len(adv_agents), args.agent_file
    )

    tasks = load_tasks(args.datasets)
    if not tasks:
        logger.error("No tasks loaded — check --dataset paths")
        sys.exit(1)
    logger.info("Total tasks: %d", len(tasks))

    # ── cost estimate (API models only) ───────────────────────────────────────
    is_local = Path(model).exists()
    if not is_local:
        total_calls = len(tasks) * len(adv_agents)
        avg_prompt = orch.estimate_prompt_tokens()
        print(
            format_batch_estimate(model, total_calls, avg_prompt, AVG_COMPLETION_TOKENS),
            file=sys.stderr,
        )
        if not args.yes:
            try:
                answer = input("\n是否继续运行？(y/N) >>> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n已取消。", file=sys.stderr)
                sys.exit(0)
            if answer not in ("y", "yes"):
                print("已取消。", file=sys.stderr)
                sys.exit(0)
    else:
        logger.info("Local model path detected — skipping cost estimate")

    # ── evaluate each agent ───────────────────────────────────────────────────
    all_results: list[dict[str, Any]] = []

    for i, agent_data in enumerate(adv_agents):
        fitness_info = (
            f"  fitness={agent_data['fitness']:.4f}" if "fitness" in agent_data else ""
        )
        adv_name = inject_agent(orch._registry, agent_data)
        logger.info(
            "[%d/%d] Evaluating: %s%s", i + 1, len(adv_agents), adv_name, fitness_info
        )

        metrics = evaluate(orch, tasks, adv_name, max_workers)
        all_results.append(metrics)

        logger.info(
            "  -> selection_rate=%.1f%%  select@1=%.1f%%  select@3=%.1f%%",
            metrics["selection_rate"] * 100,
            metrics["select@1"] * 100,
            metrics["select@3"] * 100,
        )

        # remove before next agent to avoid cross-contamination
        orch._registry._agents.pop(adv_name, None)

    # ── report ────────────────────────────────────────────────────────────────
    _print_results(all_results, args.datasets)

    # usage summary (mirrors run.py style)
    usage = orch.usage
    if usage.num_calls > 0:
        from orchestrator.cost import estimate_cost
        cost = estimate_cost(model, usage.total_prompt_tokens, usage.total_completion_tokens)
        print("=" * 60, file=sys.stderr)
        print("  Execution Summary", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"  LLM calls:         {usage.num_calls}", file=sys.stderr)
        print(f"  Prompt tokens:     {usage.total_prompt_tokens:,}", file=sys.stderr)
        print(f"  Completion tokens: {usage.total_completion_tokens:,}", file=sys.stderr)
        print(f"  Cost (est.):       ¥{cost.total_cost:.4f}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    if args.output:
        Path(args.output).write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
