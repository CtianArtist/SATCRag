"""Freeze the approved benchmark and split it once into a development set and a held-out test set.

Freezing validates every item, then writes <eval dir>/frozen/benchmark.json, dev.json, test.json
and MANIFEST.json (hashes, seed, per-item strata), marks them read-only, and refuses to overwrite
an existing freeze. Verification re-checks the hashes and that the recorded seed and strata
still reproduce the split; it needs neither the corpus nor the model.

The split: question type is an exact quota (dev gets its proportional share of every type).
Within those quotas, `candidates` random splits are drawn from the seed and the first one whose
dev set best matches the whole benchmark on style, season, single- versus multi-target and
lexical-overlap band is kept. Only item metadata is used; no retrieval result is ever consulted,
so the split cannot be shaped toward a retriever. (Reproduction relies on CPython's
random.Random.sample, unchanged since Python 3.11; the file hashes are the authoritative check.)
"""

import json
import logging
import random
import stat
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from sexandrag.artifacts import file_sha256
from sexandrag.config import EvaluationConfig, PathsConfig
from sexandrag.errors import ArtifactMissingError, EvaluationError, FrozenBenchmarkError
from sexandrag.evaluation.schema import parse_item
from sexandrag.evaluation.validate import Episodes, lexical_overlap, overlap_band, validate_raw_items

log = logging.getLogger(__name__)
BALANCE_KEYS = ("style", "seasons", "targets", "overlap")
FROZEN_NAMES = ("benchmark.json", "dev.json", "test.json")
Strata = dict[str, dict[str, Any]]


def item_strata(raw: Mapping[str, Any], episodes: Episodes) -> dict[str, Any]:
    """The attributes the split balances: type, style, seasons touched, target count and overlap band."""
    item = parse_item(raw)
    band = overlap_band(lexical_overlap(item, episodes)).split()[0]
    return {
        "question_type": item.question_type,
        "style": item.style,
        "seasons": sorted({t.season for t in item.targets}),
        "targets": "multi" if len(item.targets) > 1 else "single",
        "overlap": band,
    }


def type_quotas(strata: Strata, dev_size: int, rng: random.Random) -> dict[str, int]:
    """Dev places per question type: the proportional share, with leftover places drawn at random."""
    counts = Counter(s["question_type"] for s in strata.values())
    exact = {t: n * dev_size / len(strata) for t, n in counts.items()}
    quotas = {t: int(x) for t, x in exact.items()}
    tiebreak = {t: rng.random() for t in sorted(counts)}
    order = sorted(counts, key=lambda t: (-round(exact[t] - quotas[t], 9), tiebreak[t]))
    for t in order[: dev_size - sum(quotas.values())]:
        quotas[t] += 1
    return quotas


def attribute_counts(ids: Iterable[str], strata: Strata, key: str) -> Counter[Any]:
    """How often each value of one attribute occurs among the given items (lists count each value)."""
    counts: Counter[Any] = Counter()
    for item_id in ids:
        value = strata[item_id][key]
        counts.update(value if isinstance(value, list) else [value])
    return counts


def imbalance(dev_ids: Sequence[str], strata: Strata) -> float:
    """Distance between the dev set's attribute mix and its proportional share (0 = perfectly balanced)."""
    share = len(dev_ids) / len(strata)
    total = 0.0
    for key in BALANCE_KEYS:
        overall, dev = attribute_counts(strata, strata, key), attribute_counts(dev_ids, strata, key)
        total += sum(abs(dev[value] - share * n) for value, n in overall.items())
    return round(total, 9)


def choose_dev(strata: Strata, seed: int, dev_size: int, candidates: int) -> tuple[list[str], float]:
    """The most balanced of `candidates` seeded random splits that meet the per-type quotas."""
    if not 0 < dev_size < len(strata):
        raise EvaluationError(f"the dev set size must be between 1 and {len(strata) - 1}", actual=dev_size)
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not cryptography
    quotas = type_quotas(strata, dev_size, rng)
    by_type: defaultdict[str, list[str]] = defaultdict(list)
    for item_id in sorted(strata):
        by_type[strata[item_id]["question_type"]].append(item_id)
    best: list[str] = []
    best_score: float | None = None
    for _ in range(candidates):
        dev = sorted(i for t in sorted(by_type) for i in rng.sample(by_type[t], quotas.get(t, 0)))
        score = imbalance(dev, strata)
        if best_score is None or score < best_score:
            best, best_score = dev, score
    return best, float(best_score if best_score is not None else 0.0)


def dump(items: Sequence[Mapping[str, Any]]) -> str:
    """The exact bytes written for an eval file."""
    return json.dumps(list(items), indent=2, ensure_ascii=False) + "\n"


def write_read_only(path: Path, text: str) -> None:
    """Write a frozen file and remove its write permission (git does not keep this bit; hashes do the guarding)."""
    path.write_text(text, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def freeze(
    source: Path,
    paths: PathsConfig,
    evaluation: EvaluationConfig,
    episodes: Episodes,
    lines_sha256: str,
    corpus_sha256: str,
) -> dict[str, Any]:
    """Validate the approved items, split them, and write the frozen files plus manifest (once)."""
    out_dir = paths.frozen_dir
    if (out_dir / "MANIFEST.json").exists():
        raise FrozenBenchmarkError(
            f"{out_dir} already holds a frozen benchmark; it is never overwritten",
            recovery="to freeze a new benchmark version on purpose, move the old folder away first",
        )
    raw_items = json.loads(source.read_text(encoding="utf-8"))
    problems = validate_raw_items(raw_items, episodes)
    if problems:
        raise FrozenBenchmarkError(f"not freezing: {len(problems)} invalid item(s): {problems}")
    approved = [{**item, "status": "approved"} for item in raw_items]
    strata = {item["id"]: item_strata(item, episodes) for item in approved}
    dev_ids, score = choose_dev(strata, evaluation.split_seed, evaluation.dev_size, evaluation.split_candidates)
    parts = {
        "benchmark.json": approved,
        "dev.json": [i for i in approved if i["id"] in dev_ids],
        "test.json": [i for i in approved if i["id"] not in dev_ids],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        write_read_only(out_dir / name, dump(part))
    manifest = {
        "benchmark_version": 1,
        "frozen_on": date.today().isoformat(),
        "approval": "project owner approved freezing the reviewed items once they all validate",
        "source": {"file": source.as_posix(), "sha256": file_sha256(source)},
        "corpus": {"raw_csv_sha256": corpus_sha256, "lines_jsonl_sha256": lines_sha256},
        "files": {name: {"sha256": file_sha256(out_dir / name), "items": len(part)} for name, part in parts.items()},
        "split": {
            "seed": evaluation.split_seed,
            "dev_size": evaluation.dev_size,
            "candidates": evaluation.split_candidates,
            "balanced_on": list(BALANCE_KEYS),
            "imbalance": score,
            "dev_ids": dev_ids,
            "strata": strata,
        },
        "use": {
            "dev.json": "failure analysis, tuning, reranker selection, RRF and BM25 changes",
            "test.json": "held out: never used for tuning; run once the retrieval approach is final",
        },
    }
    write_read_only(out_dir / "MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    log.info(
        "froze %d items: %d dev, %d test (imbalance %.3f)",
        len(approved),
        len(dev_ids),
        len(approved) - len(dev_ids),
        score,
    )
    return manifest


def read_manifest(frozen_dir: Path) -> dict[str, Any]:
    """The frozen benchmark's manifest."""
    path = frozen_dir / "MANIFEST.json"
    if not path.is_file():
        raise ArtifactMissingError(f"no frozen benchmark manifest at {path}", recovery="restore eval/frozen/ from git")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrozenBenchmarkError(f"{path} is not valid JSON", recovery="restore eval/frozen/ from git") from exc
    return dict(manifest)


def verify_frozen(frozen_dir: Path) -> list[str]:
    """Problems with a frozen benchmark: changed or missing files, a split that no longer reproduces, or leaks."""
    manifest = read_manifest(frozen_dir)
    problems = []
    for name, info in manifest["files"].items():
        path = frozen_dir / name
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif file_sha256(path) != info["sha256"]:
            problems.append(f"{name} changed since freezing (sha256 {file_sha256(path)}, recorded {info['sha256']})")
    if problems:
        return problems
    split = manifest["split"]
    dev_ids, _ = choose_dev(split["strata"], split["seed"], split["dev_size"], split["candidates"])
    if dev_ids != split["dev_ids"]:
        problems.append("the recorded seed and strata no longer reproduce the dev set")
    ids = {
        name: [i["id"] for i in json.loads((frozen_dir / name).read_text(encoding="utf-8"))] for name in FROZEN_NAMES
    }
    if sorted(ids["dev.json"]) != sorted(split["dev_ids"]) or set(ids["dev.json"]) & set(ids["test.json"]):
        problems.append("dev.json and test.json do not match the recorded split")
    if sorted(ids["dev.json"] + ids["test.json"]) != sorted(ids["benchmark.json"]):
        problems.append("dev.json and test.json do not partition benchmark.json")
    return problems


def require_intact(frozen_dir: Path) -> None:
    """Raise FrozenBenchmarkError unless the frozen benchmark verifies."""
    problems = verify_frozen(frozen_dir)
    if problems:
        log.error("frozen benchmark verification failed: %s", "; ".join(problems))
        raise FrozenBenchmarkError(
            "the frozen benchmark does not match its manifest",
            actual="; ".join(problems),
            recovery="restore eval/frozen/ from git (`git checkout -- eval/frozen`); frozen files are never edited",
        )
    log.info("frozen benchmark verified: hashes match and the split reproduces")
