"""Defense 7 — Privileged Planner (CaMeL simplified).

Strip *all* self-declared free-text fields from agent profiles before showing
to the orchestrator. The orchestrator only sees agent ``name`` and the bare
list of skill ``name`` (no skill description, no agent description, no tags
that could carry payload).

Closes the IPI attack surface entirely (description = empty), at the cost of
losing semantic-match information.
"""
from __future__ import annotations

from .base import DefendedOrchestrator
from orchestrator.agent_registry import AgentRegistry


class PrivilegedPlanner(DefendedOrchestrator):
    name = "privileged_planner"

    def render_registry(self, user_query: str,
                        registry: AgentRegistry) -> str:
        # Show only name + skill name (NO description, NO tags, NO examples).
        # This is the "verified capability schema" the registry vouches for.
        lines = []
        for agent in registry.agents:
            skill_names = ", ".join(s.name for s in agent.skills)
            lines.append(
                f"- **{agent.name}** ({agent.file}): "
                f"verified_capabilities = [{skill_names}]"
            )
        return "\n".join(lines)
