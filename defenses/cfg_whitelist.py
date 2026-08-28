"""Defense 6 — CFG-based Whitelist (ControlValve simplified).

Run a planner LLM on the user query *without showing any agent profile*. The
planner returns a list of capability strings (skill names). Only agents whose
declared skills overlap with this whitelist enter the orchestrator's view.

Key property: capability planning never reads adversarial profile text, so
prompt-injection in profiles cannot affect the candidate set.
"""
from __future__ import annotations
import json
import logging
import re
from copy import deepcopy

from orchestrator.agent_registry import AgentRegistry
from orchestrator.llm_client import call_llm

from .base import DefendedOrchestrator

logger = logging.getLogger("defense.cfg_whitelist")

_PLAN_PROMPT = """You are a security-aware planner. Given ONLY the user's task
below, list the abstract agent capabilities (skill names) needed to complete
it. Do NOT consider any agent pool. Use lower-case snake_case for skill names.

User task: {q}

Return ONLY a JSON list of capability strings. No prose, no markdown. Example:
["data_analysis", "visualization"]"""


def _parse_caps(raw: str) -> list[str]:
    """Best-effort: extract a JSON list from possibly-noisy LLM output."""
    raw = raw.strip()
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Find the first [...] block
    match = re.search(r"\[[^\[\]]*\]", raw, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group(0))
            return [str(c).strip().lower() for c in arr if c]
        except Exception:
            pass
    # Fallback: split by newlines / commas
    return [c.strip().strip('"\'').lower()
            for c in re.split(r"[,\n]", raw) if c.strip()]


class CFGWhitelist(DefendedOrchestrator):
    name = "cfg_whitelist"

    def filter_registry(self, user_query: str,
                        registry: AgentRegistry) -> AgentRegistry:
        # Plan capabilities WITHOUT looking at any agent profile
        try:
            content, _ = call_llm(
                self.judge_model,
                [{"role": "user", "content": _PLAN_PROMPT.format(q=user_query)}],
                temperature=0.0, max_tokens=256,
                api_base=self.judge_api_base,
            )
            needed = _parse_caps(content)
        except Exception as exc:
            logger.warning("CFG planner failed (%s) — falling back to no-filter", exc)
            return registry

        if not needed:
            logger.warning("CFG planner returned no capabilities — skipping filter")
            return registry

        # Match-by-skill-name (case-insensitive substring on either side)
        needed_lower = [c.lower() for c in needed]
        kept_agents = {}
        for name, agent in registry._agents.items():
            agent_skills_lower = [s.name.lower() for s in agent.skills]
            agent_skill_text = " ".join(agent_skills_lower)
            # Pass if ANY needed cap appears in ANY agent skill name (or vice-versa)
            if any(c in agent_skill_text or
                   any(c in s or s in c for s in agent_skills_lower)
                   for c in needed_lower):
                kept_agents[name] = agent

        if not kept_agents:
            logger.warning(
                "CFG filter would remove all agents (caps=%s) — keeping all to avoid empty pool",
                needed_lower,
            )
            return registry

        logger.debug("CFG filter: %d → %d agents (caps=%s)",
                     len(registry._agents), len(kept_agents), needed_lower)

        new_reg = AgentRegistry()
        new_reg._agents = kept_agents
        return new_reg
