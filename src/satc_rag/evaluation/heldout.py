"""The held-out guard: the test set only runs with an explicit, deliberate opt-in.

An eval file counts as held-out when it IS the frozen test file (by path or SHA-256), or when it
contains any test item (by id or by question text), so copies, subsets and renamed items are
caught too. Without --allow-heldout such a run is refused with HeldOutSetError; with it, the run
goes ahead and is logged and recorded as a held-out run.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from satc_rag.artifacts import file_sha256
from satc_rag.config import PathsConfig
from satc_rag.errors import HeldOutSetError
from satc_rag.evaluation.schema import EvalItem

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeldOutFacts:
    """What identifies the frozen test set."""

    path: Path
    sha256: str
    ids: frozenset[str]
    questions: frozenset[str]


def held_out_facts(paths: PathsConfig) -> HeldOutFacts | None:
    """The frozen test set's identifiers, or None when no benchmark has been frozen."""
    if not paths.manifest_file.is_file() or not paths.test_file.is_file():
        return None
    manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
    test_items = json.loads(paths.test_file.read_text(encoding="utf-8"))
    return HeldOutFacts(
        path=paths.test_file.resolve(),
        sha256=manifest["files"]["test.json"]["sha256"],
        ids=frozenset(item["id"] for item in test_items),
        questions=frozenset(" ".join(item["question"].split()).casefold() for item in test_items),
    )


def held_out_reasons(eval_file: Path, items: Sequence[EvalItem], facts: HeldOutFacts) -> list[str]:
    """Why this eval file touches the held-out test set (empty when it does not)."""
    reasons = []
    if eval_file.resolve() == facts.path:
        reasons.append("it is the frozen test file")
    elif file_sha256(eval_file) == facts.sha256:
        reasons.append("it is a byte-identical copy of the frozen test file")
    shared_ids = sorted({item.id for item in items} & facts.ids)
    if shared_ids:
        reasons.append(f"it contains {len(shared_ids)} held-out item id(s), e.g. {', '.join(shared_ids[:3])}")
    shared_questions = sum(1 for item in items if " ".join(item.question.split()).casefold() in facts.questions)
    if shared_questions and not shared_ids:
        reasons.append(f"it contains {shared_questions} held-out question(s) under other ids")
    return reasons


def guard_held_out(eval_file: Path, items: Sequence[EvalItem], paths: PathsConfig, allow_heldout: bool) -> bool:
    """Refuse held-out data without the explicit opt-in; return True when this is an allowed held-out run."""
    facts = held_out_facts(paths)
    if facts is None:
        log.debug("no frozen benchmark at %s; nothing to guard", paths.frozen_dir)
        return False
    reasons = held_out_reasons(eval_file, items, facts)
    if not reasons:
        return False
    if not allow_heldout:
        log.error("refused an attempt to evaluate on held-out test data (%s)", "; ".join(reasons))
        raise HeldOutSetError(
            f"refusing to evaluate {eval_file}: {'; '.join(reasons)}",
            recovery="evaluate on the development set (`satc-rag eval`). The held-out test set runs only once the "
            "retrieval approach is final, with `satc-rag eval --split test --allow-heldout`",
        )
    log.warning("HELD-OUT TEST SET RUN (explicitly allowed): %s. Do not tune on these results.", "; ".join(reasons))
    return True
