"""Defense 1 — PromptGuard 2 / equivalent BERT-class injection classifier.

Loads a HuggingFace text-classification pipeline that returns
INJECTION / SAFE per text. Used as a **registry admission filter**: scan each
agent's name + description + skill names, and if the classifier flags any
field as injection, the agent is removed from the pool that the orchestrator
sees.

Model resolution priority:
  1. ``meta-llama/Llama-Prompt-Guard-2-86M``  (gated; requires HF auth)
  2. ``protectai/deberta-v3-base-prompt-injection-v2``  (public, used here)
"""
from __future__ import annotations
import logging
from copy import deepcopy

from orchestrator.agent_registry import AgentRegistry
from .base import DefendedOrchestrator

logger = logging.getLogger("defense.promptguard")

# Model resolution paths (cache-aware)
_MODEL_CANDIDATES = [
    "meta-llama/Llama-Prompt-Guard-2-86M",
    "protectai/deberta-v3-base-prompt-injection-v2",
]


class PromptGuard(DefendedOrchestrator):
    """Registry-admission BERT classifier defense."""
    name = "promptguard"
    threshold: float = 0.5      # min injection confidence to drop the agent
    max_seq: int = 512           # truncate per-call

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from transformers import pipeline
        self._guard = None
        for repo in _MODEL_CANDIDATES:
            try:
                self._guard = pipeline(
                    "text-classification", model=repo,
                    truncation=True, max_length=self.max_seq, device=-1,
                )
                logger.info("PromptGuard loaded: %s", repo)
                break
            except Exception as exc:
                logger.warning("PromptGuard cannot load %s: %s", repo, str(exc)[:120])
        if self._guard is None:
            raise RuntimeError("PromptGuard: no classifier model available")

    def _is_injection(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        try:
            result = self._guard(text)[0]
        except Exception as exc:
            logger.warning("PromptGuard inference failed: %s", str(exc)[:120])
            return False
        label = result.get("label", "").upper()
        score = float(result.get("score", 0.0))
        # Both label naming conventions:
        #   meta-llama PromptGuard 2: "INJECTION" / "BENIGN"
        #   protectai deberta-v3:     "INJECTION" / "SAFE"
        if "INJECTION" in label and score >= self.threshold:
            return True
        return False

    def filter_registry(self, user_query: str,
                        registry: AgentRegistry) -> AgentRegistry:
        kept = {}
        for name, agent in registry._agents.items():
            # Build text per agent: name + description + skill names
            agent_text = (
                f"{agent.name}\n{agent.description}\n"
                + "\n".join(s.name for s in agent.skills)
            )
            if self._is_injection(agent_text):
                logger.info("PromptGuard blocked: %s", name)
                continue
            kept[name] = agent

        if not kept:
            logger.warning("PromptGuard would empty the pool — keeping all to avoid break")
            return registry

        new_reg = AgentRegistry()
        new_reg._agents = kept
        return new_reg
