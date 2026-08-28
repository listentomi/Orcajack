"""Embedding-MELON — drop-in MELON variant using description-embedding distance.

Same mask-vs-real comparison as classic MELON, but instead of comparing
agent names (binary in/out), compare the cosine distance between the real
agent's *description embedding* and each mask-attractor's embedding. Only
block if max_similarity ≥ threshold.

Goal: keep MELON's high attack recall, but lower the false-positive blocking
of legitimate broad-capability agents whose descriptions look very different
from the attacker's persuasive copy.
"""
from __future__ import annotations
import logging

from orchestrator.schemas import OrchestratorOutput
from .melon import MELON

logger = logging.getLogger("defense.embedding_melon")


class EmbeddingMELON(MELON):
    name = "embedding_melon"
    similarity_threshold: float = 0.75   # cosine ≥ this → "same kind of agent"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from sentence_transformers import SentenceTransformer
        # Reuse the Phase-1 embedding model (already in available_models.json)
        local_path = (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        try:
            self._encoder = SentenceTransformer(local_path)
        except Exception:
            logger.warning("Local sentence-transformer not found, downloading…")
            self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._desc_emb_cache: dict[str, "list"] = {}
        self._mask_attractor_descs: list[str] = []
        self._mask_attractor_embs = None

    def _get_desc(self, name: str) -> str:
        agent = self.orch._registry.get(name)
        if agent is None:
            return name
        return f"{agent.name}: {agent.description}"

    def _encode(self, text: str):
        import numpy as np
        if text in self._desc_emb_cache:
            return self._desc_emb_cache[text]
        emb = self._encoder.encode(text, normalize_embeddings=True)
        self._desc_emb_cache[text] = emb
        return emb

    def _compute_mask_set(self) -> set[str]:
        # Reuse parent logic + cache attractor descriptions
        attracted = super()._compute_mask_set()
        descs = [self._get_desc(name) for name in attracted]
        self._mask_attractor_descs = descs
        if descs:
            import numpy as np
            self._mask_attractor_embs = self._encoder.encode(
                descs, normalize_embeddings=True,
            )
        return attracted

    def post_check_assignments(self, user_query: str,
                               output: OrchestratorOutput) -> list[bool]:
        import numpy as np
        self._compute_mask_set()
        if self._mask_attractor_embs is None or len(self._mask_attractor_embs) == 0:
            return [True] * len(output.assignments)

        decisions = []
        for ass in output.assignments:
            agent_desc = self._get_desc(ass.selected_agent.agent_name)
            agent_emb = self._encoder.encode(agent_desc, normalize_embeddings=True)
            # Cosine similarity (already-normalized embeddings → dot product)
            sims = self._mask_attractor_embs @ agent_emb
            max_sim = float(sims.max()) if len(sims) else 0.0
            decisions.append(max_sim < self.similarity_threshold)
        return decisions
