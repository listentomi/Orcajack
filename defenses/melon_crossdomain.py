"""MELON-CrossDomain — mask queries are REAL tasks from other domains.

Improves over generic-mask MELON variants by using real, well-formed user
tasks from *different* domains as the masked queries. If the orchestrator
picks the same agent for a finance query AND for a travel query, that is a
stronger injection signal than picking the same agent for finance + a vague
"help me with my task" placeholder.

Behavior:
  - At init, the eval harness passes a list of mask task strings (sampled
    from other-domain task files).
  - For each mask task, run the orchestrator on the FULL registry (with
    poisoned profile present) and record the agents picked.
  - Keep both a set view (for cheap membership check) AND the embedded
    agent-description vectors (for cosine comparison).
  - For each real assignment: if its agent appears in the set OR its agent
    description has cosine > threshold to any mask attractor → BLOCK.
"""
from __future__ import annotations
import logging

from orchestrator.schemas import OrchestratorOutput
from .base import DefendedOrchestrator

logger = logging.getLogger("defense.melon_crossdomain")


class MELONCrossDomain(DefendedOrchestrator):
    """MELON variant whose mask queries come from other-domain task pools."""
    name = "melon_crossdomain"
    similarity_threshold: float = 0.80
    use_embedding: bool = True   # set False for pure name-set matching

    # Set externally by eval harness before run() is called
    mask_task_pool: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Take a per-instance copy so different agents don't share state
        self.mask_task_pool = list(self.__class__.mask_task_pool)

        if self.use_embedding:
            from sentence_transformers import SentenceTransformer
            local_path = (
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            try:
                self._encoder = SentenceTransformer(local_path)
            except Exception:
                self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._mask_set: set[str] | None = None
        self._mask_descs: list[str] = []
        self._mask_embs = None

    def _agent_signature(self, name: str) -> str:
        agent = self.orch._registry.get(name)
        if agent is None: return name
        return f"{agent.name}: {agent.description}"

    def _compute_mask(self) -> tuple[set[str], list[str]]:
        if self._mask_set is not None:
            return self._mask_set, self._mask_descs
        attracted: set[str] = set()
        for q in self.mask_task_pool:
            try:
                out = self.orch.run(q)
                for ass in out.assignments:
                    attracted.add(ass.selected_agent.agent_name)
            except Exception as exc:
                logger.warning("MELON-CD mask run failed (%s): %s", q[:60], str(exc)[:120])
        self._mask_set = attracted
        self._mask_descs = [self._agent_signature(n) for n in sorted(attracted)]
        if self.use_embedding and self._mask_descs:
            self._mask_embs = self._encoder.encode(self._mask_descs, normalize_embeddings=True)
        logger.info("MELON-CD mask_set: %d agents, masks=%d queries",
                    len(attracted), len(self.mask_task_pool))
        return self._mask_set, self._mask_descs

    def post_check_assignments(self, user_query: str,
                               output: OrchestratorOutput) -> list[bool]:
        mask_set, _ = self._compute_mask()
        decisions = []
        for ass in output.assignments:
            name = ass.selected_agent.agent_name
            # Cheap path: name set membership
            if name in mask_set:
                decisions.append(False)   # block
                continue
            # Embedding path: catch agents with similar description to attractors
            if self.use_embedding and self._mask_embs is not None and len(self._mask_embs):
                sig = self._agent_signature(name)
                emb = self._encoder.encode(sig, normalize_embeddings=True)
                max_sim = float((self._mask_embs @ emb).max())
                if max_sim > self.similarity_threshold:
                    decisions.append(False)
                    continue
            decisions.append(True)
        return decisions
