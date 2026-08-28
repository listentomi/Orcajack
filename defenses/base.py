"""Defense base interface + NoDefense baseline."""
from __future__ import annotations
import threading
from typing import Any
from copy import deepcopy

from orchestrator.orchestrator import Orchestrator
from orchestrator.agent_registry import AgentRegistry
from orchestrator.schemas import OrchestratorOutput


class DefendedOrchestrator:
    """Wraps an ``Orchestrator`` and applies a defense layer.

    Subclasses override one or more of:
      - ``render_registry``: rewrite the registry text injected into the prompt
      - ``system_prompt_extra``: append text to the system prompt
      - ``filter_registry``: drop agents from registry before orchestrator sees them
      - ``run``: replace the entire run logic (e.g., MELON's double-run)
      - ``post_check_assignments``: per-assignment allow/block decision

    The default implementation is a transparent passthrough = NoDefense.
    """

    name: str = "no_defense"

    # Process-wide lock. ``run`` monkey-patches the shared ``orchestrator.prompts``
    # module functions and mutates the shared ``orch._registry._agents`` while the
    # (blocking) LLM call happens. That critical section is NOT thread-safe across
    # concurrent ``run`` calls, so we serialize it. Real parallelism for the
    # defense grid should come from separate processes per (orchestrator × domain
    # × defense) cell, not in-process threads.
    _prompt_lock = threading.Lock()

    def __init__(self, orch: Orchestrator, judge_model: str | None = None,
                 judge_api_base: str | None = None) -> None:
        self.orch = orch
        # Optional independent LLM endpoint for judges (alignment/task-shield
        # use this). Defaults to the orchestrator's own model.
        self.judge_model = judge_model or orch._model
        self.judge_api_base = judge_api_base or orch._api_base

    # ── pool-level hooks ────────────────────────────────────────────────

    def filter_registry(self, user_query: str,
                        registry: AgentRegistry) -> AgentRegistry:
        """Return a (possibly filtered) registry. Default: passthrough."""
        return registry

    def render_registry(self, user_query: str,
                        registry: AgentRegistry) -> str:
        """Return text block for the user-prompt. Default: standard render."""
        return registry.to_prompt_text()

    def system_prompt_extra(self) -> str:
        """Optional addendum appended to the orchestrator system prompt."""
        return ""

    # ── selection-level hooks ───────────────────────────────────────────

    def post_check_assignments(self, user_query: str,
                               output: OrchestratorOutput
                               ) -> list[bool]:
        """Return a list[bool] same length as output.assignments. False → block."""
        return [True] * len(output.assignments)

    # ── main ────────────────────────────────────────────────────────────

    def run(self, task: str) -> OrchestratorOutput:
        """Run orchestrator with this defense. Returns post-defense output.

        Blocked assignments are replaced with a sentinel agent name
        ``__BLOCKED_BY_DEFENSE__`` so downstream code can recognize them.
        """
        # The whole patch→run→unpatch section touches process-global state
        # (the ``orchestrator.prompts`` functions) and the shared registry, so it
        # must not interleave with another defense ``run`` on another thread.
        with DefendedOrchestrator._prompt_lock:
            # 1) Optionally filter the registry view the orchestrator will see
            original_agents = self.orch._registry._agents
            try:
                view = self.filter_registry(task, self.orch._registry)
                if view is not self.orch._registry:
                    self.orch._registry._agents = view._agents
                # 2) Override the prompt rendering for this call only
                self._patch_prompt_render(task)
                # 3) Run normally
                output = self.orch.run(task)
            finally:
                self.orch._registry._agents = original_agents
                self._unpatch_prompt_render()

        # 4) Post-check each assignment
        decisions = self.post_check_assignments(task, output)
        for ass, allow in zip(output.assignments, decisions):
            if not allow:
                ass.selected_agent.agent_name = "__BLOCKED_BY_DEFENSE__"
                ass.selected_agent.agent_file = "__blocked__.json"
                ass.selected_agent.match_reason = (
                    f"Blocked by defense {self.name}"
                )
        return output

    # ── prompt patching helpers ─────────────────────────────────────────

    def _patch_prompt_render(self, task: str) -> None:
        """Monkey-patch orchestrator's prompt render so render_registry +
        system_prompt_extra take effect for this call only."""
        from orchestrator import prompts
        self._orig_format_system = prompts.format_system_prompt
        self._orig_format_user = prompts.format_user_prompt

        rendered_registry = self.render_registry(task, self.orch._registry)
        sys_extra = self.system_prompt_extra()

        def patched_system(json_schema: str) -> str:
            base = self._orig_format_system(json_schema)
            if sys_extra:
                return base + "\n\n" + sys_extra
            return base

        def patched_user(task_arg: str, registry_text: str) -> str:
            # Override the registry text with our defense-rendered version
            return self._orig_format_user(task_arg, rendered_registry)

        prompts.format_system_prompt = patched_system
        prompts.format_user_prompt = patched_user

    def _unpatch_prompt_render(self) -> None:
        from orchestrator import prompts
        if hasattr(self, "_orig_format_system"):
            prompts.format_system_prompt = self._orig_format_system
        if hasattr(self, "_orig_format_user"):
            prompts.format_user_prompt = self._orig_format_user


class NoDefense(DefendedOrchestrator):
    """Baseline: passthrough orchestrator — used for ASR sanity check."""
    name = "no_defense"
