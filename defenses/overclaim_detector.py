"""Pool-side defense — Overclaim Detector (registration-stage description sanitizer).

Rationale: the paper's own ablation shows persuasive free-text is worth ~+60pp of
routing rate. This registration-stage defense strips persuasive / self-promotional
spans from every agent's description BEFORE the registry text is assembled, leaving
only capability-bearing sentences. It reuses the persuasion-marker lexicon from
the persuasive-language ablation, so the defense and the ablation stay consistent.

Query-independent: the sanitized registry depends only on the pool, so it is
computed once and cached across all tasks (O(1) per pool, not O(tasks)).

Hook: ``render_registry`` (requires the Priority-0a prompt-render fix to take
effect; before that fix this hook was a silent no-op).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from orchestrator.agent_registry import AgentRegistry
from .base import DefendedOrchestrator

# Reuse the exact persuasion lexicon used by the ablation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.ablate_persuasive import _PERSUASIVE_MARKERS  # noqa: E402


def _strip_persuasive_sentences(text: str) -> str:
    """Drop sentences containing a persuasion marker; keep capability sentences."""
    if not text:
        return text
    # Split into sentences on ., !, ? boundaries (keep it simple and robust).
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        sl = s.lower()
        if any(m in sl for m in _PERSUASIVE_MARKERS):
            continue
        kept.append(s)
    cleaned = " ".join(kept).strip()
    # If everything was stripped, fall back to a neutral placeholder so the agent
    # is not accidentally erased (which would be an unfair "block").
    return cleaned or "(description withheld by overclaim filter)"


class OverclaimDetector(DefendedOrchestrator):
    name = "overclaim_detector"

    def _sanitized_text(self, registry: AgentRegistry) -> str:
        sig = tuple(sorted(registry._agents.keys()))
        cache = getattr(self, "_cache", None)
        if cache and cache[0] == sig:
            return cache[1]
        lines = []
        for agent in registry.agents:
            clean_desc = _strip_persuasive_sentences(agent.description)
            skill_tags = "; ".join(
                f"{s.name} [{', '.join(s.tags)}]" for s in agent.skills
            )
            lines.append(
                f"- **{agent.name}** ({agent.file}): {clean_desc}  "
                f"Skills: {skill_tags}"
            )
        text = "\n".join(lines)
        self._cache = (sig, text)
        return text

    def render_registry(self, user_query: str, registry: AgentRegistry) -> str:
        return self._sanitized_text(registry)
