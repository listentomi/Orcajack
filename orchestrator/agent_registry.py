"""Auto-discover and load agent definitions from agents/*.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SkillInfo:
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    source: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class AgentInfo:
    name: str
    description: str
    skills: list[SkillInfo] = field(default_factory=list)
    file: str = ""  # e.g. "ml-research-scientist.json"


class AgentRegistry:
    """In-memory registry of all available agents loaded from JSON files."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentInfo] = {}

    def load_dir(self, agents_dir: str | Path) -> None:
        agents_dir = Path(agents_dir)
        if not agents_dir.is_dir():
            raise FileNotFoundError(f"Agents directory not found: {agents_dir}")

        for fp in sorted(agents_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                skills = [
                    SkillInfo(
                        name=s.get("name", ""),
                        description=s.get("description", ""),
                        tags=s.get("tags", []),
                        source=s.get("source", ""),
                        examples=s.get("examples", []),
                    )
                    for s in data.get("skills", [])
                ]
                agent = AgentInfo(
                    name=data["name"],
                    description=data.get("description", ""),
                    skills=skills,
                    file=fp.name,
                )
                self._agents[agent.name] = agent
            except Exception as exc:
                logger.warning("Skipping %s: %s", fp.name, exc)

        logger.info("Loaded %d agents from %s", len(self._agents), agents_dir)

    @property
    def agents(self) -> list[AgentInfo]:
        return list(self._agents.values())

    def get(self, name: str) -> AgentInfo | None:
        return self._agents.get(name)

    def to_prompt_text(self) -> str:
        """Format the registry as a concise text block for injection into prompts."""
        lines: list[str] = []
        for agent in self._agents.values():
            skill_tags = []
            for s in agent.skills:
                skill_tags.append(f"{s.name} [{', '.join(s.tags)}]")
            lines.append(
                f"- **{agent.name}** ({agent.file}): "
                f"{agent.description}  "
                f"Skills: {'; '.join(skill_tags)}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._agents)
