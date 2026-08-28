# Skills

`skills/<domain>/<slug>/SKILL.md` backs the capabilities that the benign agents in
`agents/` declare. `build_skill_catalog.py` reads each file's YAML frontmatter
`description` and asks an LLM for routing tags, producing `results/skill_catalog.json`.

## What is here

| Domain | Skills | Origin |
|--------|--------|--------|
| `finance/` | 8 | authored for this paper |
| `science/` | 15 | authored for this paper |
| `travel/` | 7 | authored for this paper |

## What is not here

Agent profiles in `agents/` also reference 20 general-purpose skills under a
`basic/` domain (`source: "basic/<slug>"`). Those are **not** redistributed here:
they come from the public [Anthropic Agent Skills](https://github.com/anthropics/skills)
library and remain under their own license.

This is safe to omit. `build_skill_catalog.py` skips any skill whose `SKILL.md` is
missing and records it under `skipped_no_md`, and every agent JSON already carries
each skill's `name`, `description`, `tags` and `examples` inline — the catalog only
enriches that metadata. To reproduce our exact catalog, clone the upstream library
into `skills/basic/<slug>/SKILL.md` before running the builder.
