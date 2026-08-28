#!/usr/bin/env python3
"""Build skill_catalog.json from agents/*.json + skills/<domain>/<slug>/SKILL.md.

For each unique skill referenced by any benign agent:
  1. Resolve the SKILL.md path from the agent JSON ``skill.source`` field.
  2. Parse YAML frontmatter, extract the ``description:`` field verbatim
     (used by generate_adversarial_agent.py as the SkillInfo.description metadata).
  3. Send the full SKILL.md document to a Llama-70b-class LLM and ask for
     N kebab-case tags that summarise the skill. These tags ARE rendered
     into the orchestrator routing prompt, so this LLM call is the one
     that drives the attack's "skill_tag" signal.

Output: ``results/skill_catalog.json`` by default. Consumed by
``generate_adversarial_agent.py`` via ``--skill-catalog``.

Incremental: re-running skips skills already present unless ``--refresh``.

Typical usage (with local llama-3.3-70b on vLLM):
    python3 build_skill_catalog.py \\
        --model "openai/llama-3.3-70b-instruct" \\
        --api-base "http://localhost:8006/v1" \\
        -o results/skill_catalog.json

Or via Yunwu:
    python3 build_skill_catalog.py --model "yunwu/llama-3.3-70b-instruct"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402
from orchestrator.llm_client import call_llm_json  # noqa: E402

logger = logging.getLogger("build_skill_catalog")


# ── Prompts ──────────────────────────────────────────────────────────────────

TAG_SYSTEM = (
    "You generate descriptive tags for a skill registry consumed by a "
    "Multi-Agent System router. Given the SKILL.md document for one skill, "
    "produce 4-6 kebab-case tags that summarise three axes:\n"
    "  - the skill's domain (e.g. 'finance', 'molecular-biology', 'travel-planning')\n"
    "  - 1-3 key techniques or operations (e.g. 'ratio-analysis', 'time-series-forecasting')\n"
    "  - 1-2 typical inputs or outputs (e.g. 'tabular-data', 'pdf-reports', 'mcp-tool')\n"
    "\n"
    "Constraints:\n"
    "  - All-lowercase, hyphen-separated words. No spaces, no underscores, no punctuation.\n"
    "  - Each tag must be 2-3 words at most (combined via hyphens).\n"
    "  - Avoid filler ('skill', 'agent', 'use', 'this'); avoid pure restatements of the skill name.\n"
    "  - Diverse: do not repeat the same root word across two tags.\n"
    "  - Prefer specificity ('balance-sheet') over generic terms ('finance').\n"
    "\n"
    "Output ONLY a JSON object exactly like:\n"
    '{"tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]}\n'
    "No prose, no markdown, no preamble."
)

TAG_USER = (
    "## Skill name\n{skill_name}\n\n"
    "## SKILL.md document\n```\n{skill_md_content}\n```\n\n"
    "Generate the JSON tag object."
)


# ── Helpers ──────────────────────────────────────────────────────────────────

_NAME_NORMALIZE = re.compile(r"\s+")


def normalize_name(s: str) -> str:
    """Match the GA's normalization: lower, spaces+hyphens → underscore."""
    return _NAME_NORMALIZE.sub(" ", s.strip()).lower().replace(" ", "_").replace("-", "_")


def parse_frontmatter(md_text: str) -> dict:
    """Extract YAML frontmatter between the first two ``---`` markers.

    Robust to multi-line quoted strings (the long Anthropic 'use this skill any
    time...' patterns). Returns {} on parse failure or absent frontmatter.
    """
    parts = md_text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        loaded = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        logger.warning("frontmatter YAML parse error: %s", exc)
        return {}
    return loaded or {}


def clean_tag(tag: str) -> str | None:
    """Normalize a raw tag string to kebab-case; reject empties/garbage."""
    if not isinstance(tag, str):
        return None
    t = tag.strip().lower()
    t = t.replace("_", "-").replace(" ", "-")
    # strip any residue (commas, periods, etc.)
    t = re.sub(r"[^a-z0-9\-]", "", t)
    # collapse repeated hyphens, strip leading/trailing
    t = re.sub(r"-+", "-", t).strip("-")
    if not t or len(t) < 2:
        return None
    return t


def discover_skills(agents_dir: Path) -> dict[str, dict]:
    """Walk agents/*.json, build {normalized_name: {name, source, owner_agents}}."""
    skills: dict[str, dict] = {}
    for fp in sorted(agents_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping malformed agent %s: %s", fp.name, exc)
            continue
        agent_name = data.get("name", fp.stem)
        for s in data.get("skills", []):
            raw_name = s.get("name", "")
            if not raw_name:
                continue
            key = normalize_name(raw_name)
            entry = skills.setdefault(key, {
                "name": raw_name,
                "source": s.get("source", ""),
                "owner_agents": [],
            })
            if agent_name not in entry["owner_agents"]:
                entry["owner_agents"].append(agent_name)
            # If multiple agents disagree on source, keep first; warn.
            new_source = s.get("source", "")
            if new_source and entry["source"] and new_source != entry["source"]:
                logger.warning(
                    "Skill %r has conflicting sources across agents: %r vs %r (keeping first)",
                    key, entry["source"], new_source,
                )
    return skills


def llama_generate_tags(
    skill_name: str,
    md_content: str,
    *,
    model: str,
    api_base: str | None,
    temperature: float,
    n_tags: int,
    max_tokens: int = 256,
) -> list[str]:
    """One LLM call → cleaned, deduped, length-capped list of tags."""
    messages = [
        {"role": "system", "content": TAG_SYSTEM},
        {"role": "user", "content": TAG_USER.format(
            skill_name=skill_name,
            # cap to keep prompt under model's 8K-32K context; SKILL.md tends to be short
            skill_md_content=md_content[:8000],
        )},
    ]
    try:
        result, _usage = call_llm_json(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
        )
    except Exception as exc:
        logger.warning("LLM tag call failed for %r: %s", skill_name, exc)
        return []

    raw_tags = result.get("tags") if isinstance(result, dict) else None
    if not isinstance(raw_tags, list):
        logger.warning("LLM returned non-list tags for %r: %s", skill_name, result)
        return []

    out: list[str] = []
    seen: set[str] = set()
    for t in raw_tags:
        c = clean_tag(t)
        if c and c not in seen:
            out.append(c)
            seen.add(c)
        if len(out) >= n_tags:
            break
    return out


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build skill_catalog.json (description from SKILL.md + LLM-generated tags).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--agents-dir", default="agents",
                    help="Directory of benign agent JSON files (default: agents)")
    ap.add_argument("--skills-dir", default="skills",
                    help="Root of skills/<domain>/<slug>/SKILL.md tree (default: skills)")
    ap.add_argument("-o", "--output", default="results/skill_catalog.json",
                    help="Output JSON path (default: results/skill_catalog.json)")
    ap.add_argument("--model", default="openai/llama-3.3-70b-instruct",
                    help="Llama-70b-class model id (litellm format). Default: "
                         "openai/llama-3.3-70b-instruct (combine with --api-base "
                         "pointing to your local vLLM).")
    ap.add_argument("--api-base", default="http://localhost:8006/v1",
                    help="Override api_base for the LLM endpoint. Default: "
                         "http://localhost:8006/v1 (the local llama-3.3-70b vLLM).")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="Sampling temperature for tag generation (default 0.2).")
    ap.add_argument("--n-tags", type=int, default=5,
                    help="Tags per skill (default 5; range 4-6 enforced in the prompt).")
    ap.add_argument("--workers", type=int, default=5,
                    help="Parallel LLM workers (default 5).")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-generate tags for all skills, even those already in the catalog.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    agents_dir = Path(args.agents_dir)
    skills_root = Path(args.skills_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load existing catalog ───────────────────────────────────────────
    existing: dict[str, dict] = {}
    existing_metadata: dict = {}
    if out_path.exists() and not args.refresh:
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            existing = payload.get("skills", {})
            existing_metadata = payload.get("metadata", {})
            logger.info(
                "Loaded %d existing entries from %s (use --refresh to regenerate)",
                len(existing), out_path,
            )
        except Exception as exc:
            logger.warning("Could not parse existing catalog %s: %s", out_path, exc)

    # ── Discover skills + resolve SKILL.md ──────────────────────────────
    skills_meta = discover_skills(agents_dir)
    logger.info("Discovered %d unique skills across %s/", len(skills_meta), agents_dir)

    # Build the work list: (key, name, md_path, description_from_frontmatter, full_md_text)
    work: list[tuple[str, str, Path, str, str]] = []
    skipped_no_md: list[str] = []
    for key, meta in sorted(skills_meta.items()):
        md_path = skills_root / meta["source"] / "SKILL.md"
        if not md_path.exists():
            skipped_no_md.append(f"{key} → {md_path}")
            continue
        try:
            md_text = md_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Cannot read %s: %s", md_path, exc)
            continue
        fm = parse_frontmatter(md_text)
        desc_raw = fm.get("description", "")
        if not isinstance(desc_raw, str):
            desc_raw = json.dumps(desc_raw, ensure_ascii=False)
        description = desc_raw.strip()
        work.append((key, meta["name"], md_path, description, md_text))

    if skipped_no_md:
        logger.warning("Skipped %d skills with missing SKILL.md:\n  %s",
                       len(skipped_no_md), "\n  ".join(skipped_no_md[:10]))

    # Filter: incremental mode skips entries already in the catalog
    catalog: dict[str, dict] = dict(existing)
    if not args.refresh:
        work_filtered = [w for w in work if w[0] not in existing]
        if len(work_filtered) < len(work):
            logger.info(
                "Skipping %d skills already cached (use --refresh to override)",
                len(work) - len(work_filtered),
            )
        work = work_filtered

    logger.info("Will generate tags for %d skills via %s (api_base=%s, workers=%d)",
                len(work), args.model, args.api_base or "<default>", args.workers)

    # ── Parallel LLM calls ──────────────────────────────────────────────

    def _process(item) -> tuple[str, dict]:
        key, name, md_path, desc, content = item
        tags = llama_generate_tags(
            name, content,
            model=args.model, api_base=args.api_base,
            temperature=args.temperature, n_tags=args.n_tags,
        )
        return key, {
            "name": name,
            "skill_md_path": str(md_path),
            "description": desc,
            "tags": tags,
            "owner_agents": list(skills_meta[key]["owner_agents"]),
        }

    completed = 0
    failed: list[str] = []
    if work:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_process, w): w[0] for w in work}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    k, entry = fut.result()
                    catalog[k] = entry
                    completed += 1
                    short_desc = entry["description"][:60].replace("\n", " ")
                    logger.info(
                        "[%d/%d] %s  tags=%s  desc=%r%s",
                        completed, len(work), k, entry["tags"],
                        short_desc, "…" if len(entry["description"]) > 60 else "",
                    )
                except Exception as exc:
                    failed.append(key)
                    logger.error("[%s] worker raised: %s", key, exc)

    # ── Ensure catalog covers every discovered skill (even on LLM failure) ──
    for key, meta in skills_meta.items():
        if key in catalog:
            continue
        md_path = skills_root / meta["source"] / "SKILL.md"
        description = ""
        if md_path.exists():
            try:
                fm = parse_frontmatter(md_path.read_text(encoding="utf-8"))
                d = fm.get("description", "")
                if isinstance(d, str):
                    description = d.strip()
            except Exception:
                pass
        catalog[key] = {
            "name": meta["name"],
            "skill_md_path": str(md_path),
            "description": description,
            "tags": [],
            "owner_agents": meta["owner_agents"],
        }

    # ── Persist ─────────────────────────────────────────────────────────
    payload = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tag_model": args.model,
            "tag_api_base": args.api_base or "",
            "tag_temperature": args.temperature,
            "n_tags_target": args.n_tags,
            "n_skills": len(catalog),
            "prior_metadata": existing_metadata if existing else None,
        },
        "skills": catalog,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved %d skill entries → %s", len(catalog), out_path)

    # Summary
    n_with_tags = sum(1 for e in catalog.values() if e.get("tags"))
    n_with_desc = sum(1 for e in catalog.values() if e.get("description"))
    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Catalog ready: {out_path}", file=sys.stderr)
    print(f"  total skills:        {len(catalog)}", file=sys.stderr)
    print(f"  with non-empty tags: {n_with_tags}", file=sys.stderr)
    print(f"  with description:    {n_with_desc}", file=sys.stderr)
    if failed:
        print(f"  ⚠  failed:           {len(failed)} → {failed}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
