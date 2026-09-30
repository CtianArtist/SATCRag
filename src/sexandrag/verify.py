"""`sexandrag verify`: check every local artifact against what it should be, without changing anything.

Components (each can be selected with --only):
  corpus     the raw CSV exists and has the pinned SHA-256
  metadata   episode titles and speaker aliases load and validate
  lines      lines.jsonl exists and was parsed from the pinned corpus
  chunks     every configured chunk set matches its manifest, lines.jsonl and tokenizer
  index      every dense index matches its key, chunk ids and vector checks (incomplete builds flagged)
  model      the pinned model snapshot is present and its files match their digests (--deep: weights too)
  benchmark  the frozen eval files match their hashes and the split reproduces (needs no corpus)

A component is "missing" when its artifact has not been built yet (a fresh clone) and "failed"
when it exists but is wrong. Failures give a nonzero exit; missing items do too with --strict.
"""

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sexandrag.artifacts import ChunkConfig, chunk_configs, file_sha256
from sexandrag.chunk import ChunkSet, load_chunk_set
from sexandrag.config import Settings
from sexandrag.episodes import load_episode_titles
from sexandrag.errors import ArtifactMissingError, ModelNotAvailableError, SexAndRagError
from sexandrag.evaluation.split import verify_frozen
from sexandrag.index import index_location, load_index, rebuild_hint
from sexandrag.model import snapshot_path, verify_snapshot
from sexandrag.services import Services
from sexandrag.speakers import load_aliases

log = logging.getLogger(__name__)
COMPONENTS = ("corpus", "metadata", "lines", "chunks", "index", "model", "benchmark")


@dataclass(frozen=True)
class Check:
    """The outcome of one check."""

    component: str
    name: str
    status: str  # "ok" | "missing" | "failed" | "warning"
    detail: str = ""


def outcome(component: str, name: str, action: Callable[[], str]) -> Check:
    """Run one check, turning expected errors into a status instead of a crash."""
    try:
        return Check(component, name, "ok", action())
    except (ArtifactMissingError, ModelNotAvailableError) as exc:
        return Check(component, name, "missing", str(exc))
    except SexAndRagError as exc:
        return Check(component, name, "failed", str(exc))


def check_corpus(settings: Settings) -> list[Check]:
    """The raw CSV and its checksum."""
    raw = settings.paths.raw_csv
    if not raw.is_file():
        return [Check("corpus", raw.name, "missing", "run `sexandrag download`")]
    digest = file_sha256(raw)
    if digest != settings.corpus.raw_csv_sha256:
        return [Check("corpus", raw.name, "failed", f"sha256 {digest} != pinned {settings.corpus.raw_csv_sha256}")]
    return [Check("corpus", raw.name, "ok", f"sha256 {digest[:12]}... matches the pinned version")]


def check_metadata(settings: Settings) -> list[Check]:
    """Episode titles and speaker aliases."""
    paths = settings.paths
    return [
        outcome(
            "metadata", paths.episodes_csv.name, lambda: f"{len(load_episode_titles(paths.episodes_csv))} episodes"
        ),
        outcome(
            "metadata",
            paths.speaker_aliases.name,
            lambda: f"{len(load_aliases(paths.speaker_aliases).global_aliases)} global aliases",
        ),
    ]


def check_lines(settings: Settings) -> list[Check]:
    """lines.jsonl and the corpus it was parsed from."""
    paths = settings.paths
    if not paths.lines_jsonl.is_file():
        return [Check("lines", paths.lines_jsonl.name, "missing", "run `sexandrag parse`")]
    try:
        source = json.loads(paths.parse_report.read_text(encoding="utf-8"))["source"]["sha256"]
    except (OSError, KeyError, json.JSONDecodeError):
        return [Check("lines", paths.parse_report.name, "failed", "parse report missing or unreadable; re-run parse")]
    if source != settings.corpus.raw_csv_sha256:
        return [Check("lines", paths.lines_jsonl.name, "failed", "parsed from a different corpus; re-run parse")]
    return [Check("lines", paths.lines_jsonl.name, "ok", "parsed from the pinned corpus")]


def try_load_chunks(config: ChunkConfig, settings: Settings) -> tuple[ChunkSet | None, Check]:
    """Load one chunk set for checking: (the set or None, its check)."""
    paths = settings.paths
    try:
        chunk_set = load_chunk_set(config, paths.chunks_dir, paths.lines_jsonl)
    except ArtifactMissingError as exc:
        return None, Check("chunks", config.config_id, "missing", str(exc))
    except SexAndRagError as exc:
        return None, Check("chunks", config.config_id, "failed", str(exc))
    return chunk_set, Check("chunks", config.config_id, "ok", f"{len(chunk_set.chunks)} chunks match the manifest")


def check_chunks_and_index(settings: Settings, services: Services, components: Sequence[str]) -> list[Check]:
    """Every configured chunk set, then the dense index of each valid one."""
    paths = settings.paths
    tokenizer, revision = services.tokenizer_id(settings)
    checks: list[Check] = []
    for config in chunk_configs(settings.chunking, tokenizer, revision):
        chunk_set, chunk_check = try_load_chunks(config, settings)
        if "chunks" in components:
            checks.append(chunk_check)
        if "index" in components and chunk_set is not None:
            checks.append(check_index(chunk_set, services, settings))
    if "index" in components:
        checks += [
            Check("index", folder.name, "warning", "incomplete build left behind (safe to delete)")
            for folder in sorted(paths.index_dir.glob("dense/*/*.tmp"))
        ]
    return checks


def check_index(chunk_set: ChunkSet, services: Services, settings: Settings) -> Check:
    """One dense index, verified without loading the model."""
    identity = services.embedder_identity(settings)
    key, directory = index_location(chunk_set, identity, settings.paths.index_dir)
    name = f"{chunk_set.config.config_id}/{directory.name}"
    if not directory.is_dir():
        return Check("index", name, "missing", f"run `sexandrag index --size {chunk_set.config.size}`")
    ids = [chunk["chunk_id"] for chunk in chunk_set.chunks]

    def verify() -> str:
        index = load_index(directory, key, ids, identity["dim"], rebuild_hint(chunk_set.config))
        return f"{len(index.chunk_ids)} vectors verified against key {key[:12]}"

    return outcome("index", name, verify)


def check_model(settings: Settings, deep: bool) -> list[Check]:
    """The pinned model snapshot in the local cache."""
    spec = settings.model.spec

    def verify() -> str:
        verify_snapshot(snapshot_path(spec, settings.paths.model_cache_dir), spec, deep=deep)
        return "all pinned files match" + (" (weights hashed)" if deep else " (weights not hashed; use --deep)")

    return [outcome("model", f"{spec.repo_id}@{spec.revision[:8]}", verify)]


def check_benchmark(settings: Settings) -> list[Check]:
    """The frozen benchmark's hashes and split."""
    frozen = settings.paths.frozen_dir
    if not (frozen / "MANIFEST.json").is_file():
        return [Check("benchmark", "eval/frozen", "missing", "no frozen benchmark in this checkout")]
    problems = verify_frozen(frozen)
    if problems:
        return [Check("benchmark", "eval/frozen", "failed", "; ".join(problems))]
    return [Check("benchmark", "eval/frozen", "ok", "hashes match and the dev/test split reproduces")]


def run_checks(
    settings: Settings, services: Services, components: Sequence[str] = COMPONENTS, deep: bool = False
) -> list[Check]:
    """Run the selected checks in order and log a line per problem."""
    checks: list[Check] = []
    if "corpus" in components:
        checks += check_corpus(settings)
    if "metadata" in components:
        checks += check_metadata(settings)
    if "lines" in components:
        checks += check_lines(settings)
    if "chunks" in components or "index" in components:
        checks += check_chunks_and_index(settings, services, components)
    if "model" in components:
        checks += check_model(settings, deep)
    if "benchmark" in components:
        checks += check_benchmark(settings)
    for check in checks:
        if check.status == "failed":
            log.error("%s %s: %s", check.component, check.name, check.detail)
        elif check.status in ("missing", "warning"):
            log.warning("%s %s: %s", check.component, check.name, check.detail.splitlines()[0])
    return checks
