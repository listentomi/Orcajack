"""Defense 5 — MELON adapted for agent-selection (Zhu et al., ICML 2025).

Run the orchestrator twice:
  - Real: user_query + full (poisoned) pool → A_real
  - Masked: generic placeholder query + same pool → A_masked

If both runs select the same agent for the same subtask role, the agent's
attractiveness is driven by its profile content (potentially injection)
rather than user intent → block. Catches "always-pick-me" injections; can be
bypassed by query-conditional injections.
"""
from __future__ import annotations
import logging

from orchestrator.schemas import OrchestratorOutput

from .base import DefendedOrchestrator

logger = logging.getLogger("defense.melon")

_MASK_QUERIES = [
    "Help me with my task.",
    "Please assist with this request.",
    "I need help with the following.",
]


class MELON(DefendedOrchestrator):
    name = "melon"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Cache mask outputs across queries — registry is constant within
        # a single benchmark run, so masked output is identical for all queries
        # (modulo LLM stochasticity at temperature > 0).
        self._mask_agents: set[str] | None = None

    def _compute_mask_set(self) -> set[str]:
        """Run orchestrator once on each masked query; collect agents that
        appear (i.e., agents the pool 'attracts' regardless of query)."""
        if self._mask_agents is not None:
            return self._mask_agents
        attracted: set[str] = set()
        for q in _MASK_QUERIES:
            try:
                output = self.orch.run(q)
                for ass in output.assignments:
                    attracted.add(ass.selected_agent.agent_name)
            except Exception as exc:
                logger.warning("MELON mask run failed (%s) — skipping", exc)
        self._mask_agents = attracted
        logger.debug("MELON mask set: %d agents attracted by generic queries", len(attracted))
        return attracted

    def post_check_assignments(self, user_query: str,
                               output: OrchestratorOutput) -> list[bool]:
        mask_set = self._compute_mask_set()
        decisions = []
        for ass in output.assignments:
            name = ass.selected_agent.agent_name
            # If the agent ALSO appears in the masked-query pool attractor set,
            # its attractiveness is profile-driven → block.
            decisions.append(name not in mask_set)
        return decisions
