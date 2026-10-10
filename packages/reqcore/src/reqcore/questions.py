"""Render Jev question sets and LLM prompts from the versioned YAML files in `questions/`.

Both renderers read the same taxonomy, so Jev and the LLM baselines see identical class
definitions and examples. Every rendered artifact carries a hash of its exact content; the
hash is part of the inference cache key and of every stored prediction.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from reqcore.hashing import content_hash
from reqcore.paths import questions_dir


@cache
def load_yaml(name: str, directory: Path | None = None) -> dict[str, Any]:
    path = (directory or questions_dir()) / f"{name}.yaml"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Taxonomy:
    version: str
    leaves: tuple[str, ...]
    level1: dict[str, tuple[str, ...]]
    promise_mapping: dict[str, str]
    classes: dict[str, dict[str, str]]
    examples: dict[str, tuple[str, ...]]

    def level1_of(self, leaf: str) -> str:
        return next(group for group, leaves in self.level1.items() if leaf in leaves)

    def criterion(self, leaf: str) -> dict[str, Any]:
        """Structured criterion for one option: what it covers, what it does not, examples."""
        entry: dict[str, Any] = {"what": self.classes[leaf]["what"]}
        if not_for := self.classes[leaf].get("not_for"):
            entry["not_for"] = not_for
        if examples := self.examples.get(leaf):
            entry["examples"] = list(examples)
        return entry


def load_taxonomy(name: str = "taxonomy_v1", examples: str | None = None) -> Taxonomy:
    raw = load_yaml(name)
    example_map = load_yaml(examples)["examples"] if examples else {}
    return Taxonomy(
        version=raw["version"],
        leaves=tuple(raw["classes"]),
        level1={group: tuple(leaves) for group, leaves in raw["level1"].items()},
        promise_mapping=dict(raw["promise_mapping"]),
        classes=raw["classes"],
        examples={leaf: tuple(items) for leaf, items in example_map.items()},
    )


@dataclass(frozen=True)
class JevQuestionSet:
    version: str
    model: str
    rotation: int
    questions: dict[str, dict[str, Any]]
    hash: str


def _rotate(options: list[str], rotation: int) -> list[str]:
    shift = rotation % len(options)
    return options[shift:] + options[:shift]


def render_jev_questions(
    name: str = "jev_type_v1", rotation: int = 0, examples: str | None = None
) -> JevQuestionSet:
    """Render the Jev question map. `rotation` cyclically shifts the order of Choice options."""
    spec = load_yaml(name)
    taxonomy = load_taxonomy(spec["taxonomy"], examples)
    questions: dict[str, dict[str, Any]] = {}
    for qid, choice in spec.get("choices", {}).items():
        options = _rotate(list(choice["options"]), rotation)
        questions[qid] = {
            "type": "choice",
            "instructions": choice["instructions"],
            "criteria": {leaf: taxonomy.criterion(leaf) for leaf in options},
        }
    for qid, noul in spec.get("nouls", {}).items():
        question: dict[str, Any] = {"type": "noul", "instructions": noul["instructions"]}
        if "criteria" in noul:
            question["criteria"] = noul["criteria"]
        elif "criteria_from_taxonomy" in noul:
            question["criteria"] = {
                side: {leaf: taxonomy.classes[leaf]["what"] for leaf in leaves}
                for side, leaves in noul["criteria_from_taxonomy"].items()
            }
        questions[qid] = question
    version = f"{spec['version']}+{examples or 'noex'}+r{rotation}"
    # canonical_json sorts keys, so the option order (which the model sees) is hashed explicitly.
    option_order = {
        qid: list(q["criteria"]) for qid, q in questions.items() if q["type"] == "choice"
    }
    payload = {"model": spec["model"], "questions": questions, "option_order": option_order}
    return JevQuestionSet(version, spec["model"], rotation, questions, content_hash(payload))


@dataclass(frozen=True)
class LlmPrompt:
    version: str
    system: str
    user_template: str
    labels: tuple[str, ...]
    hash: str

    def user(self, requirement: str) -> str:
        return self.user_template.replace("{requirement}", requirement)

    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": list(self.labels)}},
            "required": ["label"],
            "additionalProperties": False,
        }


def render_llm_prompt(name: str = "llm_type_v1", examples: str | None = None) -> LlmPrompt:
    spec = load_yaml(name)
    taxonomy = load_taxonomy(spec["taxonomy"], examples)
    blocks = []
    for leaf in taxonomy.leaves:
        criterion = taxonomy.criterion(leaf)
        lines = [f"- {leaf}: {criterion['what']}"]
        if "not_for" in criterion:
            lines.append(f"  Not for: {criterion['not_for']}")
        for example in criterion.get("examples", []):
            lines.append(f'  Example: "{example}"')
        blocks.append("\n".join(lines))
    system = spec["system"].replace("{definitions}", "\n".join(blocks))
    payload = {"system": system, "user": spec["user"], "labels": list(taxonomy.leaves)}
    return LlmPrompt(
        version=f"{spec['version']}+{examples or 'noex'}",
        system=system,
        user_template=spec["user"],
        labels=taxonomy.leaves,
        hash=content_hash(payload),
    )
