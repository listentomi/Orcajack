"""Pool-side defense — Schema Whitelist (registration-stage skill-name validation).

NOTE ON NAMING (honest scoping): this is a *schema / declared-name whitelist*, NOT
behavioral capability verification. The registry stores no provider evidence,
reputation, or execution proof (``AgentInfo`` has only name/description/skills), so
we cannot verify that a declared skill is truthfully backed by behavior. What we
CAN do at registration time is check that every skill an agent declares corresponds
to a skill the registry actually knows about (the trusted skill catalog built from
the legitimate pool). An agent that invents skills outside the catalog is dropped.

Expected honest result: an ORCAJACK profile that mines *legitimate* skill atoms
(which is exactly what Stage-1 skill mining does) will largely PASS this whitelist —
demonstrating that name-level validation is insufficient and motivating real
capability verification. That negative result is the point.

Query-independent → computed once per pool and cached.
Hook: ``filter_registry`` (effective regardless of the 0a fix).
"""
from __future__ import annotations

import json
from pathlib import Path

from orchestrator.agent_registry import AgentRegistry
from .base import DefendedOrchestrator

_CATALOG_PATH = Path(__file__).resolve().parents[1] / "results" / "skill_catalog.json"


def _load_trusted_terms(path: Path = _CATALOG_PATH) -> set[str]:
    """Trusted skill-name + tag vocabulary from the catalog (lower-cased)."""
    terms: set[str] = set()
    try:
        cat = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return terms
    for entry in (cat.get("skills") or {}).values():
        if entry.get("name"):
            terms.add(str(entry["name"]).strip().lower())
        for t in entry.get("tags", []) or []:
            terms.add(str(t).strip().lower())
    return terms


class SchemaWhitelist(DefendedOrchestrator):
    name = "schema_whitelist"

    # Minimum fraction of an agent's declared skills that must be catalog-known
    # for the agent to be admitted. Below this, the agent is dropped.
    min_known_fraction: float = 0.5

    def _trusted(self) -> set[str]:
        if not hasattr(self, "_trusted_terms"):
            self._trusted_terms = _load_trusted_terms()
        return self._trusted_terms

    def _known(self, term: str, trusted: set[str]) -> bool:
        tl = term.strip().lower()
        if not tl:
            return False
        if tl in trusted:
            return True
        # substring match either direction (skill-name variants)
        return any(tl in t or t in tl for t in trusted)

    def filter_registry(self, user_query: str,
                        registry: AgentRegistry) -> AgentRegistry:
        trusted = self._trusted()
        if not trusted:  # catalog missing → fail open (no-op), don't erase pool
            return registry

        sig = tuple(sorted(registry._agents.keys()))
        cache = getattr(self, "_kept_cache", None)
        if cache and cache[0] == sig:
            kept_names = cache[1]
        else:
            kept_names = []
            for name, agent in registry._agents.items():
                skills = agent.skills or []
                if not skills:
                    kept_names.append(name)  # nothing to validate → keep
                    continue
                known = 0
                for s in skills:
                    terms = [s.name] + list(s.tags or [])
                    if any(self._known(t, trusted) for t in terms):
                        known += 1
                if known / len(skills) >= self.min_known_fraction:
                    kept_names.append(name)
            self._kept_cache = (sig, kept_names)

        view = AgentRegistry()
        view._agents = {n: registry._agents[n] for n in kept_names}
        return view
