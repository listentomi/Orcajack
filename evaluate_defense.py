#!/usr/bin/env python3
"""Evaluate adversarial agent profiles against a panel of defenses.

For each (defense × adversarial agent × task) triplet we measure whether the
adversarial agent ends up in the orchestrator's final assignments. The diff
of selection_rate vs. NoDefense baseline tells us how much each defense
actually blocks the attack.

Usage:
    python3 evaluate_defense.py \\
        -a results/v20260426_finance_R1_L3/top3_agents.json \\
        -d Datasets/finance-agent-benchmark/split_42_40_balanced/task.test.json \\
        --orchestrator-model openai/qwen2.5-7b-instruct \\
        --orchestrator-api-base http://localhost:8004/v1 \\
        --judge-model openai/qwen2.5-7b-instruct \\
        --judge-api-base http://localhost:8004/v1 \\
        --defenses no_defense spotlighting cfg_whitelist privileged_planner alignment_check task_shield melon \\
        -w 4 -y \\
        -o results/v20260429_defense_eval/finance_R1L3_qwen.json
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import threading

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.agent_registry import AgentInfo, AgentRegistry, SkillInfo
from orchestrator.orchestrator import Orchestrator

from defenses import (
    NoDefense, PromptGuard, Spotlighting, CFGWhitelist, PrivilegedPlanner,
    AlignmentCheck, TaskShield, MELON, EmbeddingMELON, MELONv2, MELONCrossDomain,
    ProfileConsistencyCheck, OverclaimDetector, ReputationPrior, SchemaWhitelist,
)

DEFENSE_REGISTRY = {
    "no_defense":         NoDefense,
    "promptguard":        PromptGuard,
    "spotlighting":       Spotlighting,
    "cfg_whitelist":      CFGWhitelist,
    "privileged_planner": PrivilegedPlanner,
    "alignment_check":    AlignmentCheck,
    "task_shield":        TaskShield,
    "melon":              MELON,
    "embedding_melon":    EmbeddingMELON,
    "melon_v2":           MELONv2,
    "melon_crossdomain":  MELONCrossDomain,
    # Pool-side / registration-stage validation defenses.
    "profile_consistency": ProfileConsistencyCheck,
    "overclaim_detector":  OverclaimDetector,
    "reputation_prior":    ReputationPrior,
    "schema_whitelist":    SchemaWhitelist,
}

logger = logging.getLogger("eval_defense")


def load_adv_agents(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def load_tasks(path: str) -> list[str]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for e in entries:
        desc = e.get("task_description") or e.get("task_inst", "")
        if desc:
            out.append(desc)
    return out


def inject(orch: Orchestrator, agent_data: dict) -> str:
    skills = [SkillInfo(name=s.get("name", ""),
                         description=s.get("description", ""),
                         tags=s.get("tags", []))
              for s in agent_data.get("skills", [])]
    fname = agent_data.get("file",
                           agent_data["name"].lower().replace(" ", "-") + ".json")
    orch._registry._agents[agent_data["name"]] = AgentInfo(
        name=agent_data["name"], description=agent_data.get("description", ""),
        skills=skills, file=fname,
    )
    return agent_data["name"]


def run_one(defense, task: str, adv_name: str) -> dict:
    try:
        out = defense.run(task)
        names = [a.selected_agent.agent_name for a in out.assignments]
        any_blocked = "__BLOCKED_BY_DEFENSE__" in names
        return {
            "selected": adv_name in names,
            "first": bool(names) and names[0] == adv_name,
            "top3": adv_name in names[:3],
            "any_blocked_by_defense": any_blocked,
        }
    except Exception as exc:
        logger.warning("[%s] task failed: %s", defense.name, str(exc)[:200])
        return {"selected": False, "first": False, "top3": False,
                "any_blocked_by_defense": False, "error": str(exc)[:200]}


def evaluate_one_defense(defense, tasks: list[str], adv_name: str,
                         workers: int) -> dict:
    hits: list[dict] = []
    lock = threading.Lock()
    counter = [0]

    def _collect(t):
        r = run_one(defense, t, adv_name)
        with lock:
            hits.append(r)
            counter[0] += 1
            if counter[0] % 5 == 0 or counter[0] == len(tasks):
                logger.info("  [%s] progress %d/%d", defense.name, counter[0], len(tasks))

    if workers <= 1:
        for t in tasks:
            _collect(t)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_collect, t) for t in tasks]
            for f in as_completed(futs):
                try: f.result()
                except Exception as exc:
                    logger.warning("worker error: %s", str(exc)[:200])

    n = len(hits)
    return {
        "defense": defense.name,
        "agent": adv_name,
        "n_tasks": n,
        "selection_rate": sum(h["selected"] for h in hits) / n if n else 0.0,
        "select@1": sum(h["first"] for h in hits) / n if n else 0.0,
        "select@3": sum(h["top3"] for h in hits) / n if n else 0.0,
        "block_rate": sum(h["any_blocked_by_defense"] for h in hits) / n if n else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("-a", "--agent-file", required=True)
    p.add_argument("-d", "--dataset", required=True,
                   help="Single test split task JSON")
    p.add_argument("--orchestrator-model", required=True)
    p.add_argument("--orchestrator-api-base", default=None)
    p.add_argument("--judge-model", default=None,
                   help="Independent judge LLM for AlignmentCheck/TaskShield/CFGWhitelist; "
                        "defaults to orchestrator model")
    p.add_argument("--judge-api-base", default=None)
    p.add_argument("--defenses", nargs="+", default=list(DEFENSE_REGISTRY.keys()),
                   choices=list(DEFENSE_REGISTRY.keys()))
    p.add_argument("--config", default="orchestrator/config.yaml")
    p.add_argument("-w", "--workers", type=int, default=4)
    p.add_argument("-y", "--yes", action="store_true")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--max-tasks", type=int, default=0,
                   help="Cap test tasks to first N (0 = all)")
    p.add_argument("--mask-task-files", nargs="*", default=[],
                   help="Other-domain test JSONs sampled to build MELONCrossDomain "
                        "mask queries (only used when 'melon_crossdomain' is in --defenses)")
    p.add_argument("--mask-task-count", type=int, default=3,
                   help="Number of mask tasks to sample from --mask-task-files")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    orch = Orchestrator(config_path=args.config)
    orch._model = args.orchestrator_model
    if args.orchestrator_api_base:
        orch._api_base = args.orchestrator_api_base
    orch._temperature = 0.0
    orch._fallback_model = ""  # disable fallback to keep behavior deterministic

    adv_agents = load_adv_agents(args.agent_file)
    tasks = load_tasks(args.dataset)
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    logger.info("Loaded %d adv agents × %d tasks × %d defenses = %d evaluations",
                len(adv_agents), len(tasks), len(args.defenses),
                len(adv_agents) * len(tasks) * len(args.defenses))

    judge_model = args.judge_model or args.orchestrator_model
    judge_api_base = args.judge_api_base or args.orchestrator_api_base

    # Sample mask tasks for MELONCrossDomain
    if args.mask_task_files:
        import random
        rng = random.Random(0)
        pool = []
        for mf in args.mask_task_files:
            for e in json.loads(Path(mf).read_text()):
                t = e.get("task_description") or e.get("task_inst","")
                if t: pool.append(t)
        rng.shuffle(pool)
        sample = pool[:args.mask_task_count]
        MELONCrossDomain.mask_task_pool = sample
        logger.info("MELONCrossDomain mask pool (%d sampled from %d total):", len(sample), len(pool))
        for q in sample:
            logger.info("  • %s", q[:100])

    all_results: list[dict] = []
    for ag_idx, agent in enumerate(adv_agents, 1):
        adv_name = inject(orch, agent)
        logger.info("[%d/%d] Adversarial agent: %s",
                    ag_idx, len(adv_agents), adv_name)
        for d_name in args.defenses:
            defense = DEFENSE_REGISTRY[d_name](
                orch, judge_model=judge_model, judge_api_base=judge_api_base,
            )
            logger.info("  Defense: %s", d_name)
            metrics = evaluate_one_defense(defense, tasks, adv_name, args.workers)
            logger.info(
                "    → selection_rate=%.1f%%  block_rate=%.1f%%",
                metrics["selection_rate"] * 100, metrics["block_rate"] * 100,
            )
            all_results.append(metrics)
        # Remove this adversarial agent before injecting the next one
        orch._registry._agents.pop(adv_name, None)

    # ── Pretty print ────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"  Defense Evaluation Results — {args.dataset}")
    print(f"  Orchestrator: {args.orchestrator_model}  Judge: {judge_model}")
    print("=" * 90)
    by_agent: dict[str, list[dict]] = {}
    for r in all_results:
        by_agent.setdefault(r["agent"], []).append(r)
    for agent_name, rows in by_agent.items():
        print(f"\nAgent: {agent_name}")
        print(f"  {'defense':<22} {'sel.rate':>10} {'@1':>8} {'@3':>8} {'block.rate':>11}")
        print("  " + "-" * 64)
        baseline = next((r for r in rows if r["defense"] == "no_defense"), None)
        for r in rows:
            sel = r["selection_rate"] * 100
            tag = ""
            if baseline and r["defense"] != "no_defense":
                drop = baseline["selection_rate"] * 100 - sel
                tag = f"  (Δ −{drop:.1f}pp)" if drop >= 0 else f"  (Δ +{-drop:.1f}pp)"
            print(f"  {r['defense']:<22} {sel:>9.1f}% {r['select@1']*100:>7.1f}%"
                  f" {r['select@3']*100:>7.1f}% {r['block_rate']*100:>10.1f}%{tag}")
    print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("Saved → %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
