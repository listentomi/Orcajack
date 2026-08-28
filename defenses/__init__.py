"""Defense modules against agent-metadata-poisoning prompt injection.

All defenses follow the same interface (DefendedOrchestrator) so the eval
harness in ``evaluate_defense.py`` can swap them in/out.

Implemented:
  - NoDefense (baseline / sanity check)
  - Spotlighting    (Defense 2; Hines et al. 2024)
  - CFGWhitelist    (Defense 6; ControlValve simplified)
  - PrivilegedPlanner (Defense 7; CaMeL simplified)
  - AlignmentCheck  (Defense 3; LlamaFirewall simplified)
  - TaskShield      (Defense 4; Jia et al. ACL 2025, agent-selection adapted)
  - MELON           (Defense 5; Zhu et al. ICML 2025, agent-selection adapted)

PromptGuard 2 (Defense 1) requires a HuggingFace model download on first use.
"""
from __future__ import annotations
from .base import DefendedOrchestrator, NoDefense
from .spotlight import Spotlighting
from .cfg_whitelist import CFGWhitelist
from .privileged_planner import PrivilegedPlanner
from .alignment_check import AlignmentCheck
from .task_shield import TaskShield
from .melon import MELON
from .embedding_melon import EmbeddingMELON
from .promptguard import PromptGuard
# Pool-side / registration-stage validation defenses.
from .profile_consistency import ProfileConsistencyCheck
from .reputation_prior import ReputationPrior
from .schema_whitelist import SchemaWhitelist

ALL_DEFENSES = [
    NoDefense,
    PromptGuard,
    Spotlighting,
    CFGWhitelist,
    PrivilegedPlanner,
    AlignmentCheck,
    TaskShield,
    MELON,
    EmbeddingMELON,
    ProfileConsistencyCheck,
    ReputationPrior,
    SchemaWhitelist,
]

__all__ = [
    "DefendedOrchestrator", "NoDefense",
    "PromptGuard", "Spotlighting", "CFGWhitelist", "PrivilegedPlanner",
    "AlignmentCheck", "TaskShield", "MELON", "EmbeddingMELON",
    "ProfileConsistencyCheck", "ReputationPrior", "SchemaWhitelist",
    "ALL_DEFENSES",
]
