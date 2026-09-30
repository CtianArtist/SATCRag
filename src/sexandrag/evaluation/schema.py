"""The eval-item schema: a question, its scored evidence groups (ALL / ANY), and premise-only evidence.

Scored evidence is a list of required groups. The answer is fully supported when every group is
satisfied:
  * mode "all": every target in the group must be retrieved (e.g. a sequence question's anchor
    and answer, or the passages of a multi-episode arc);
  * mode "any": the targets are alternatives; retrieving one of them satisfies the group.
A retrieved chunk is relevant when it overlaps any scored target. Premise evidence ("evidence",
role "premise") only lets a reviewer check the question's wording and is never scored.

Three equivalent spellings are accepted, one per item:
  1. a single target: season, episode, source_row_start, source_row_end on the item itself;
  2. "targets": [...] plus "requires_all_targets": true (one "all" group) or false (one "any"
     group); the flag is required when there are several targets;
  3. "required_groups": [{"mode": "all" | "any", "targets": [...]}, ...] (the general form).
Every target needs explicit rows: an episode alone never makes a chunk relevant. Invalid items
raise EvaluationSchemaError when they are loaded, never later during scoring.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sexandrag.errors import EvaluationSchemaError
from sexandrag.provenance import Span

QUESTION_TYPES = ("who_said", "scene_fact", "character", "episode", "sequence", "arc")
STYLES = ("lexical", "paraphrase", "relationship", "event", "multi_evidence", "metadata")
MODES = ("all", "any")
SPAN_FIELDS = ("season", "episode", "source_row_start", "source_row_end")
ROW_FIELDS = ("source_row_start", "source_row_end")
EVIDENCE_FORMS = ("targets", "required_groups")


@dataclass(frozen=True)
class EvidenceGroup:
    """Targets that must all be retrieved (mode "all") or of which one suffices (mode "any")."""

    mode: str
    targets: tuple[Span, ...]

    def requirements(self) -> tuple[tuple[Span, ...], ...]:
        """The group as independent requirements: one per target for "all", one shared for "any"."""
        return tuple((target,) for target in self.targets) if self.mode == "all" else (self.targets,)


@dataclass(frozen=True)
class EvalItem:
    """One validated eval item."""

    id: str
    question: str
    groups: tuple[EvidenceGroup, ...]
    premise: tuple[Span, ...] = ()
    question_type: str | None = None
    style: str | None = None
    expected_answer: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def targets(self) -> tuple[Span, ...]:
        """Every scored target, in the order the item lists them."""
        return tuple(target for group in self.groups for target in group.targets)

    @property
    def requirements(self) -> tuple[tuple[Span, ...], ...]:
        """Independent pieces of evidence the answer needs; each is satisfied by any of its spans."""
        return tuple(req for group in self.groups for req in group.requirements())

    @property
    def needs_several(self) -> bool:
        """True when the answer needs more than one independent piece of evidence (AllTargetsHit applies)."""
        return len(self.requirements) > 1

    def is_relevant(self, chunk: Mapping[str, Any]) -> bool:
        """True when the chunk overlaps any scored target (premise evidence never counts)."""
        return any(target.hits(chunk) for target in self.targets)


def fail(item_id: str, problem: str, expected: Any = None, actual: Any = None) -> EvaluationSchemaError:
    """An EvaluationSchemaError for one item."""
    return EvaluationSchemaError(
        f"eval item {item_id!r}: {problem}",
        expected=expected,
        actual=actual,
        recovery="fix the item (see docs/evaluation.md for the schema)",
    )


def is_int(value: Any) -> bool:
    """True for real integers (bool excluded)."""
    return isinstance(value, int) and not isinstance(value, bool)


def parse_span(obj: Any, item_id: str, where: str) -> Span:
    """Validate one target or evidence span."""
    if not isinstance(obj, Mapping):
        raise fail(item_id, f"{where} must be an object", actual=obj)
    if not all(key in obj for key in ("season", "episode")) or not all(is_int(obj[k]) for k in ("season", "episode")):
        raise fail(item_id, f"{where} needs integer season and episode", actual={k: obj.get(k) for k in SPAN_FIELDS})
    if obj.get("source_row_start") is None:
        raise fail(
            item_id,
            f"{where} has no source rows; relevance must come from answer-bearing rows, never the episode alone",
            expected="source_row_start and source_row_end",
        )
    start, end = obj["source_row_start"], obj.get("source_row_end", obj["source_row_start"])
    if not (is_int(start) and is_int(end) and 0 <= start <= end):
        raise fail(item_id, f"{where} rows must be integers with 0 <= start <= end", actual=(start, end))
    for key in ("expected_quote", "role"):
        if obj.get(key) is not None and not isinstance(obj[key], str):
            raise fail(item_id, f"{where} {key} must be a string", actual=obj[key])
    return Span(obj["season"], obj["episode"], start, end, obj.get("expected_quote"), obj.get("role"))


def parse_target_list(value: Any, item_id: str, where: str) -> tuple[Span, ...]:
    """A non-empty list of target spans."""
    if not isinstance(value, list) or not value:
        raise fail(item_id, f"{where} must be a non-empty list of targets", actual=value)
    return tuple(parse_span(obj, item_id, f"{where}[{i}]") for i, obj in enumerate(value))


def parse_group(obj: Any, item_id: str, where: str) -> EvidenceGroup:
    """One {"mode": ..., "targets": [...]} group."""
    if not isinstance(obj, Mapping) or set(obj) != {"mode", "targets"}:
        raise fail(item_id, f"{where} must be exactly {{mode, targets}}", actual=obj)
    if obj["mode"] not in MODES:
        raise fail(item_id, f"{where} mode must be 'all' or 'any'", actual=obj["mode"])
    targets = parse_target_list(obj["targets"], item_id, f"{where}.targets")
    if obj["mode"] == "any" and len(targets) < 2:
        raise fail(item_id, f"{where} is an 'any' group with a single target; list alternatives or use 'all'")
    return EvidenceGroup(obj["mode"], targets)


def parse_groups(raw: Mapping[str, Any], item_id: str) -> tuple[EvidenceGroup, ...]:
    """The item's scored evidence in any of the three accepted spellings, as groups."""
    forms = [key for key in EVIDENCE_FORMS if key in raw]
    inline = [key for key in (*SPAN_FIELDS, "expected_quote") if key in raw]
    if len(forms) > 1 or (forms and inline):
        raise fail(
            item_id,
            "give scored evidence in exactly one way",
            expected="a single target, 'targets' or 'required_groups'",
            actual=forms + inline,
        )
    if "requires_all_targets" in raw and forms != ["targets"]:
        raise fail(item_id, "requires_all_targets only applies to a 'targets' list")
    if "required_groups" in raw:
        groups = raw["required_groups"]
        if not isinstance(groups, list) or not groups:
            raise fail(item_id, "required_groups must be a non-empty list", actual=groups)
        return tuple(parse_group(obj, item_id, f"required_groups[{i}]") for i, obj in enumerate(groups))
    if "targets" in raw:
        targets = parse_target_list(raw["targets"], item_id, "targets")
        flag = raw.get("requires_all_targets")
        if len(targets) > 1 and not isinstance(flag, bool):
            raise fail(item_id, "several targets need an explicit requires_all_targets (true = all, false = any)")
        if len(targets) == 1 and flag is not None:
            raise fail(item_id, "requires_all_targets is only for several targets")
        return (EvidenceGroup("all" if flag is not False else "any", targets),)
    return (EvidenceGroup("all", (parse_span(raw, item_id, "the item's target"),)),)


def parse_premise(raw: Mapping[str, Any], item_id: str) -> tuple[Span, ...]:
    """Premise-only evidence: validated spans with role "premise", never scored."""
    evidence = raw.get("evidence", [])
    if not isinstance(evidence, list):
        raise fail(item_id, "evidence must be a list of premise spans", actual=evidence)
    spans = tuple(parse_span(obj, item_id, f"evidence[{i}]") for i, obj in enumerate(evidence))
    wrong = [span.label for span in spans if span.role != "premise"]
    if wrong:
        raise fail(
            item_id, "every evidence span must have role 'premise' (scored spans belong in the targets)", actual=wrong
        )
    return spans


def optional_label(raw: Mapping[str, Any], key: str, allowed: Sequence[str], item_id: str) -> str | None:
    """A question_type or style label, checked against the known values when present."""
    value = raw.get(key)
    if value is not None and value not in allowed:
        raise fail(item_id, f"unknown {key}", expected=", ".join(allowed), actual=value)
    return value


def parse_item(raw: Any) -> EvalItem:
    """Validate one raw eval item and return it with normalized evidence groups."""
    if not isinstance(raw, Mapping):
        raise fail("?", "an eval item must be a JSON object", actual=type(raw).__name__)
    item_id = raw.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise fail(str(item_id), "needs a non-empty string 'id'")
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise fail(item_id, "needs a non-empty 'question'")
    answer = raw.get("expected_answer")
    if answer is not None and not isinstance(answer, str):
        raise fail(item_id, "expected_answer must be a string", actual=answer)
    return EvalItem(
        id=item_id,
        question=question,
        groups=parse_groups(raw, item_id),
        premise=parse_premise(raw, item_id),
        question_type=optional_label(raw, "question_type", QUESTION_TYPES, item_id),
        style=optional_label(raw, "style", STYLES, item_id),
        expected_answer=answer,
        raw=raw,
    )


def parse_items(raw_items: Any, source: str) -> list[EvalItem]:
    """Validate a list of raw items, reporting every invalid item and duplicate id together."""
    if not isinstance(raw_items, list) or not raw_items:
        raise EvaluationSchemaError(f"{source} must hold a non-empty JSON list of eval items")
    items, problems = [], []
    for raw in raw_items:
        try:
            items.append(parse_item(raw))
        except EvaluationSchemaError as exc:
            problems.append(exc.message + (f" (expected {exc.expected}; got {exc.actual})" if exc.expected else ""))
    duplicates = sorted(i for i, n in Counter(item.id for item in items).items() if n > 1)
    if duplicates:
        problems.append(f"duplicate ids: {', '.join(duplicates)}")
    if problems:
        shown = "\n".join(f"  - {p}" for p in problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise EvaluationSchemaError(
            f"{source}: {len(problems)} schema problem(s):\n{shown}{more}",
            recovery="fix the items (see docs/evaluation.md); nothing was run",
        )
    return items


def load_items(path: Path) -> list[EvalItem]:
    """Read and validate an eval file; any problem is reported before anything runs."""
    if not path.is_file():
        raise EvaluationSchemaError(f"eval file not found: {path}", recovery="check the --eval-file path")
    try:
        raw_items = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationSchemaError(
            f"{path} is not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})"
        ) from exc
    return parse_items(raw_items, str(path))
