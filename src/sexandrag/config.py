"""One validated configuration layer; the defaults reproduce the frozen baseline.

Settings are built from three layers, later ones winning:
  1. the defaults below (the configuration every published number is based on),
  2. an optional TOML file: --config, $SEXANDRAG_CONFIG, or <root>/sexandrag.toml,
  3. command-line options for the few things that change per run (root, verbosity).

Unknown keys, wrong types and contradictory values raise ConfigurationError before any work
starts. Nothing falls back silently: an unsupported model, revision, pooling, device or chunk
setting is an error, never a quiet substitution.

The TOML file uses one table per section, e.g.:

    [paths]
    index_dir = "/mnt/cache/sexandrag-index"   # relative paths are resolved against the root
    [retrieval]
    hybrid_candidates = 30
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from sexandrag.errors import ConfigurationError, ModelConfigurationError

ENV_ROOT = "SEXANDRAG_ROOT"
ENV_CONFIG = "SEXANDRAG_CONFIG"
CONFIG_FILENAME = "sexandrag.toml"
COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DEVICES = ("cpu", "cuda")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
SPECIAL_TOKENS = 2  # [CLS] and [SEP]: the embedder sees a chunk's tokens plus these


@dataclass(frozen=True)
class ModelSpec:
    """Everything pinned about one supported embedding model; any mismatch at load time is an error.

    `files` maps each required file (relative to the snapshot) to its digest: a git blob SHA-1
    (40 hex) for small files or a SHA-256 (64 hex) for large LFS files, exactly as the Hugging
    Face cache names them.
    """

    repo_id: str
    revision: str
    max_tokens: int
    dim: int
    pooling: Mapping[str, Any]
    modules: tuple[str, ...]
    files: Mapping[str, str]
    tokenizer_file: str = "tokenizer.json"

    def identity(self) -> dict[str, Any]:
        """What changes the vectors; part of every dense cache key (never includes device or batch size)."""
        return {
            "model": self.repo_id,
            "revision": self.revision,
            "max_tokens": self.max_tokens,
            "dim": self.dim,
            "normalized": True,
            "pooling": dict(self.pooling),
        }


BGE_M3 = ModelSpec(
    repo_id="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    max_tokens=8192,
    dim=1024,
    pooling=MappingProxyType({"embedding_dimension": 1024, "pooling_mode": "cls", "include_prompt": True}),
    modules=(
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    ),
    files=MappingProxyType(
        {
            "config.json": "e6eda1c72da8f9dc30fdd9b69c73d35af3b7a7ad",
            "config_sentence_transformers.json": "1fba91c78a6c8e17227058ab6d4d3acb5d8630a9",
            "modules.json": "952a9b81c0bfd99800fabf352f69c7ccd46c5e43",
            "sentence_bert_config.json": "0140ba1eac83a3c9b857d64baba91969d988624b",
            "1_Pooling/config.json": "9bd85925f325e25246d94c4918dc02ab98f2a1b7",
            "special_tokens_map.json": "b1879d702821e753ffe4245048eee415d54a9385",
            "tokenizer_config.json": "dc69ac559dcba2694012009aaa108c614541789a",
            "tokenizer.json": "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08",
            "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
            "pytorch_model.bin": "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38",
        }
    ),
)
SUPPORTED_MODELS: Mapping[tuple[str, str], ModelSpec] = MappingProxyType({(BGE_M3.repo_id, BGE_M3.revision): BGE_M3})


@dataclass(frozen=True)
class CorpusConfig:
    """The source dataset and the parser's one judgement call."""

    kaggle_dataset: str = "snapcrack/every-sex-and-the-city-script/versions/3"
    raw_csv_sha256: str = "3572c0e7851600080bafb40ae691696983caa4946f5079e3093b19d23d734047"
    # A row like "- Hello? - Carrie, it's Stanford." holds two turns. The dataset's label can
    # belong to either, so by default neither inherits it (speaker=None). True gives it to the first.
    split_turns_keep_label_on_first: bool = False
    unknown_speaker: str = "(unknown)"  # how speaker=None lines are shown in chunk text


@dataclass(frozen=True)
class ModelConfig:
    """Which pinned embedding model to use and how to run it (device and batch size never change vectors' identity)."""

    repo_id: str = BGE_M3.repo_id
    revision: str = BGE_M3.revision
    device: str = "cpu"
    batch_size: int = 4

    @property
    def spec(self) -> ModelSpec:
        """The pinned specification for this model and revision."""
        try:
            return SUPPORTED_MODELS[(self.repo_id, self.revision)]
        except KeyError:
            supported = ", ".join(f"{r}@{v}" for r, v in SUPPORTED_MODELS)
            raise ModelConfigurationError(
                "unsupported embedding model or revision",
                expected=supported,
                actual=f"{self.repo_id}@{self.revision}",
                recovery="use the pinned model, or add a ModelSpec with verified file digests to sexandrag.config",
            ) from None


@dataclass(frozen=True)
class SceneConfig:
    """Optional heuristic scene grouping (metadata only, never ground truth)."""

    enabled: bool = True
    window: int = 6  # lines compared on each side of a gap
    max_similarity: float = 0.25  # rarity-weighted speaker-set Jaccard at or below this marks a turnover
    min_lines: int = 6  # no inferred scene shorter than this many lines
    transition_pattern: str = (  # narration-style openers that usually start a new scene
        r"^(?:(?:later|earlier) that (?:day|night|evening|afternoon|morning|week)\b"
        r"|later on\b|meanwhile\b"
        r"|that (?:same )?(?:day|night|evening|afternoon|morning|weekend)\b"
        r"|the (?:next|following) (?:day|night|morning|evening|afternoon|week|weekend)\b"
        r"|next (?:morning|day|night|evening|week)\b"
        r"|(?:a|one|two|three|four|five|a few|several) (?:hours?|days?|weeks?|months?) later\b"
        r"|across town\b"
        r"|(?:on )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday) "
        r"(?:morning|afternoon|evening|night)\b)"
    )


@dataclass(frozen=True)
class ChunkingConfig:
    """Token-based chunking: sizes are counted in the embedding model's own tokens."""

    size: int = 512  # default chunk set for single-size commands
    overlap: int = 64
    configs: tuple[tuple[int, int], ...] = ((256, 32), (512, 64), (1024, 128))  # the primary experiment


@dataclass(frozen=True)
class RetrievalConfig:
    """BM25 and fusion parameters of the baseline."""

    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # rank_bm25 sets the IDF of terms in more than half the chunks to epsilon x (average IDF). Its
    # default 0.25 gives "the"/"you"/"Carrie" an IDF of ~1.3-1.6 on this corpus, more than terms in
    # just under half the chunks (IDF near 0). 0.0 clamps them to zero, so IDF never rises with
    # document frequency (the textbook max(0, idf) form).
    bm25_epsilon: float = 0.0
    rrf_k: int = 60  # Reciprocal Rank Fusion constant: score = sum 1 / (rrf_k + rank)
    hybrid_candidates: int = 30  # how deep each component ranking goes into the fusion
    default_k: int = 8  # results shown by `sexandrag search`


@dataclass(frozen=True)
class EvaluationConfig:
    """Metric cut-offs, stored depth, and the one-time dev/test split."""

    k_values: tuple[int, ...] = (1, 5, 10)  # Hit@k and Recall@k; MRR is computed over the top max(k)
    all_targets_k: tuple[int, ...] = (5, 10)  # AllTargetsHit@k, for items needing several pieces of evidence
    compare_k: int = 5  # cut-off used to label per-query wins and losses between retrievers
    depth: int = 50  # ranked results stored per query for failure analysis
    split_seed: int = 20261001
    dev_size: int = 20
    split_candidates: int = 20000


PATH_KEYS = (
    "raw_csv",
    "episodes_csv",
    "speaker_aliases",
    "processed_dir",
    "index_dir",
    "eval_dir",
    "results_dir",
    "model_cache_dir",
)


@dataclass(frozen=True)
class PathsConfig:
    """Where inputs and artifacts live. Everything defaults to a location under the project root."""

    root: Path
    raw_csv: Path
    episodes_csv: Path
    speaker_aliases: Path
    processed_dir: Path
    index_dir: Path
    eval_dir: Path
    results_dir: Path
    model_cache_dir: Path | None = None  # None: the Hugging Face default (~/.cache/huggingface/hub)

    @classmethod
    def under(cls, root: Path) -> PathsConfig:
        """The standard layout of a checkout rooted at `root`."""
        return cls(
            root=root,
            raw_csv=root / "data" / "raw" / "SATC_all_lines.csv",
            episodes_csv=root / "data" / "meta" / "episodes.csv",
            speaker_aliases=root / "data" / "meta" / "speaker_aliases.json",
            processed_dir=root / "data" / "processed",
            index_dir=root / "index",
            eval_dir=root / "eval",
            results_dir=root / "results",
        )

    @property
    def lines_jsonl(self) -> Path:
        """Parsed dialogue lines, one JSON record per turn."""
        return self.processed_dir / "lines.jsonl"

    @property
    def parse_report(self) -> Path:
        """What the parser changed and dropped, and why."""
        return self.processed_dir / "parse_report.json"

    @property
    def scenes_jsonl(self) -> Path:
        """Optional inferred scenes."""
        return self.processed_dir / "scenes_inferred.jsonl"

    @property
    def chunks_dir(self) -> Path:
        """Chunk sets: one .jsonl plus one .manifest.json per setting."""
        return self.processed_dir / "chunks"

    @property
    def frozen_dir(self) -> Path:
        """The frozen benchmark and its split (read-only)."""
        return self.eval_dir / "frozen"

    @property
    def benchmark_file(self) -> Path:
        """All approved benchmark items."""
        return self.frozen_dir / "benchmark.json"

    @property
    def dev_file(self) -> Path:
        """Development set: failure analysis and tuning."""
        return self.frozen_dir / "dev.json"

    @property
    def test_file(self) -> Path:
        """Held-out test set: never used for tuning."""
        return self.frozen_dir / "test.json"

    @property
    def manifest_file(self) -> Path:
        """Hashes, split seed and strata of the frozen benchmark."""
        return self.frozen_dir / "MANIFEST.json"

    @property
    def runs_dir(self) -> Path:
        """One directory per benchmark run."""
        return self.results_dir / "runs"


SECTION_TYPES: dict[str, type] = {
    "corpus": CorpusConfig,
    "model": ModelConfig,
    "chunking": ChunkingConfig,
    "scenes": SceneConfig,
    "retrieval": RetrievalConfig,
    "evaluation": EvaluationConfig,
}


@dataclass(frozen=True)
class Settings:
    """The complete, validated configuration for one invocation."""

    paths: PathsConfig
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    scenes: SceneConfig = field(default_factory=SceneConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    log_level: str = "INFO"

    @classmethod
    def defaults(cls, root: Path) -> Settings:
        """The baseline configuration for a checkout at `root` (validated)."""
        return cls(paths=PathsConfig.under(root)).validate()

    def validate(self) -> Settings:
        """Return self, or raise ConfigurationError listing every problem found."""
        problems = (
            corpus_problems(self.corpus)
            + model_problems(self.model)
            + chunking_problems(self.chunking, self.model)
            + scene_problems(self.scenes)
            + retrieval_problems(self.retrieval)
            + evaluation_problems(self.evaluation)
        )
        if self.log_level not in LOG_LEVELS:
            problems.append(f"log_level must be one of {', '.join(LOG_LEVELS)}, not {self.log_level!r}")
        if problems:
            raise ConfigurationError(
                "invalid configuration:\n" + "\n".join(f"  - {p}" for p in problems),
                recovery="fix the listed settings in your config file (see `sexandrag config show` for the defaults)",
            )
        return self

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form (paths as strings) for run records and `sexandrag config show`."""
        record = asdict(self)
        record["paths"] = {key: None if value is None else str(value) for key, value in record["paths"].items()}
        record["model"]["spec_identity"] = self.model.spec.identity()
        return record


def corpus_problems(corpus: CorpusConfig) -> list[str]:
    """Contradictions in the corpus settings."""
    problems = []
    if not SHA256_RE.fullmatch(corpus.raw_csv_sha256):
        problems.append("corpus.raw_csv_sha256 must be 64 lowercase hex characters")
    if not corpus.unknown_speaker.strip():
        problems.append("corpus.unknown_speaker must not be blank")
    return problems


def model_problems(model: ModelConfig) -> list[str]:
    """Unsupported model, unpinned revision, device or batch size."""
    problems = []
    if not COMMIT_SHA_RE.fullmatch(model.revision):
        problems.append(
            f"model.revision must be a full 40-character commit hash (branches, tags and pull-request refs "
            f"such as refs/pr/N are refused), not {model.revision!r}"
        )
    elif (model.repo_id, model.revision) not in SUPPORTED_MODELS:
        supported = ", ".join(f"{r}@{v}" for r, v in SUPPORTED_MODELS)
        problems.append(f"model {model.repo_id}@{model.revision} is not a supported pinned model ({supported})")
    if model.device not in DEVICES:
        problems.append(f"model.device must be one of {', '.join(DEVICES)}, not {model.device!r}")
    if model.batch_size < 1:
        problems.append("model.batch_size must be at least 1")
    return problems


def chunk_setting_problems(size: int, overlap: int, max_tokens: int) -> list[str]:
    """Why one (size, overlap) chunk setting is invalid for a model context, if it is."""
    problems = []
    if size < 1:
        problems.append(f"chunk size must be positive (got {size})")
    if not 0 <= overlap < max(size, 1):
        problems.append(f"chunk overlap must satisfy 0 <= overlap < size (got size {size}, overlap {overlap})")
    if size + SPECIAL_TOKENS > max_tokens:
        problems.append(
            f"chunk size {size} plus {SPECIAL_TOKENS} special tokens exceeds the model context of {max_tokens} "
            "tokens; chunks would be truncated"
        )
    return problems


def chunking_problems(chunking: ChunkingConfig, model: ModelConfig) -> list[str]:
    """Invalid or contradictory chunk settings."""
    key = (model.repo_id, model.revision)
    max_tokens = SUPPORTED_MODELS[key].max_tokens if key in SUPPORTED_MODELS else BGE_M3.max_tokens
    problems = []
    if not chunking.configs:
        problems.append("chunking.configs must list at least one (size, overlap) pair")
    for size, overlap in chunking.configs:
        problems += chunk_setting_problems(size, overlap, max_tokens)
    sizes = [size for size, _ in chunking.configs]
    if len(sizes) != len(set(sizes)):
        problems.append("chunking.configs has two settings with the same size")
    if (chunking.size, chunking.overlap) not in chunking.configs:
        problems.append(
            f"the default chunk setting ({chunking.size}, {chunking.overlap}) is not one of chunking.configs"
        )
    return problems


def scene_problems(scenes: SceneConfig) -> list[str]:
    """Invalid scene-heuristic settings."""
    problems = []
    if scenes.window < 1 or scenes.min_lines < 1:
        problems.append("scenes.window and scenes.min_lines must be at least 1")
    if not 0.0 <= scenes.max_similarity <= 1.0:
        problems.append("scenes.max_similarity must be between 0 and 1")
    try:
        re.compile(scenes.transition_pattern)
    except re.error as exc:
        problems.append(f"scenes.transition_pattern is not a valid regular expression: {exc}")
    return problems


def retrieval_problems(retrieval: RetrievalConfig) -> list[str]:
    """Out-of-range BM25 or fusion parameters."""
    problems = []
    if retrieval.bm25_k1 <= 0:
        problems.append("retrieval.bm25_k1 must be positive")
    if not 0.0 <= retrieval.bm25_b <= 1.0:
        problems.append("retrieval.bm25_b must be between 0 and 1")
    if retrieval.bm25_epsilon < 0:
        problems.append("retrieval.bm25_epsilon must not be negative")
    for name in ("rrf_k", "hybrid_candidates", "default_k"):
        if getattr(retrieval, name) < 1:
            problems.append(f"retrieval.{name} must be at least 1")
    return problems


def evaluation_problems(evaluation: EvaluationConfig) -> list[str]:
    """Contradictory metric cut-offs or split settings."""
    problems = []
    k_values = evaluation.k_values
    if not k_values or any(k < 1 for k in k_values) or list(k_values) != sorted(set(k_values)):
        problems.append("evaluation.k_values must be positive, strictly increasing cut-offs")
    elif not set(evaluation.all_targets_k) <= set(k_values):
        problems.append("evaluation.all_targets_k must be a subset of evaluation.k_values")
    elif evaluation.compare_k not in k_values:
        problems.append("evaluation.compare_k must be one of evaluation.k_values")
    elif evaluation.depth < max(k_values):
        problems.append("evaluation.depth must be at least the largest k in evaluation.k_values")
    if evaluation.dev_size < 1 or evaluation.split_candidates < 1:
        problems.append("evaluation.dev_size and evaluation.split_candidates must be at least 1")
    return problems


def resolve_root(root: Path | None = None) -> Path:
    """The project root: --root, else $SEXANDRAG_ROOT, else the current directory."""
    env = os.environ.get(ENV_ROOT)
    candidate = (root or (Path(env) if env else Path.cwd())).expanduser().resolve()
    if not candidate.is_dir():
        raise ConfigurationError(
            "the project root is not a directory",
            actual=str(candidate),
            recovery=f"run from the repository checkout, or pass --root / set {ENV_ROOT}",
        )
    return candidate


def resolve_config_file(root: Path, config_file: Path | None = None) -> Path | None:
    """The TOML file to read: --config, else $SEXANDRAG_CONFIG, else <root>/sexandrag.toml if present."""
    env = os.environ.get(ENV_CONFIG)
    if config_file or env:
        path = (config_file or Path(str(env))).expanduser()
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise ConfigurationError("config file not found", actual=str(path), recovery="check the --config path")
        return path
    default = root / CONFIG_FILENAME
    return default if default.is_file() else None


def read_config_file(path: Path) -> dict[str, Any]:
    """Parse a TOML config file, reporting syntax errors with the file name."""
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"cannot parse config file {path}: {exc}", recovery="fix the TOML syntax") from exc


def coerce(value: Any, default: Any, name: str) -> Any:
    """Convert a TOML value to the type of the setting's default, refusing anything else."""
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, float):
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        value = float(value) if ok else value
    elif isinstance(default, str):
        ok = isinstance(value, str)
    elif isinstance(default, tuple):
        nested = bool(default) and isinstance(default[0], tuple)
        ok = isinstance(value, list | tuple) and all(
            isinstance(v, list | tuple) and all(type(x) is int for x in v) if nested else type(v) is int for v in value
        )
        value = tuple(tuple(v) for v in value) if ok and nested else (tuple(value) if ok else value)
    else:
        ok = False
    if not ok:
        raise ConfigurationError(
            f"setting {name} has the wrong type",
            expected=type(default).__name__,
            actual=f"{value!r} ({type(value).__name__})",
        )
    return value


def apply_section(section: Any, values: Mapping[str, Any], name: str) -> Any:
    """Return a copy of one config section with `values` applied; unknown keys are errors."""
    known = {f.name for f in fields(section)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ConfigurationError(
            f"unknown setting(s) in [{name}]: {', '.join(unknown)}",
            expected=", ".join(sorted(known)),
            recovery="check the spelling against `sexandrag config show`",
        )
    updates = {key: coerce(value, getattr(section, key), f"{name}.{key}") for key, value in values.items()}
    return replace(section, **updates)


def apply_paths(paths: PathsConfig, values: Mapping[str, Any]) -> PathsConfig:
    """Return a copy of the path settings with `values` applied (relative paths resolve against the root)."""
    unknown = sorted(set(values) - set(PATH_KEYS))
    if unknown:
        raise ConfigurationError(f"unknown setting(s) in [paths]: {', '.join(unknown)}", expected=", ".join(PATH_KEYS))
    updates = {}
    for key, value in values.items():
        if not isinstance(value, str) or not value:
            raise ConfigurationError(f"setting paths.{key} must be a non-empty path string", actual=repr(value))
        path = Path(value).expanduser()
        updates[key] = path if path.is_absolute() else paths.root / path
    return replace(paths, **updates)


def build_settings(root: Path, data: Mapping[str, Any]) -> Settings:
    """Apply parsed TOML data on top of the defaults for `root` (not yet validated)."""
    allowed = {"paths", "log_level", *SECTION_TYPES}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(
            f"unknown config section(s): {', '.join(unknown)}", expected=", ".join(sorted(allowed))
        )
    settings = Settings(paths=apply_paths(PathsConfig.under(root), data.get("paths", {})))
    for name in SECTION_TYPES:
        values = data.get(name, {})
        if not isinstance(values, Mapping):
            raise ConfigurationError(f"[{name}] must be a table of settings")
        settings = replace(settings, **{name: apply_section(getattr(settings, name), values, name)})
    if "log_level" in data:
        settings = replace(settings, log_level=str(data["log_level"]).upper())
    return settings


def load_settings(root: Path | None = None, config_file: Path | None = None, log_level: str | None = None) -> Settings:
    """Resolve the root, read the optional config file, apply overrides, and validate."""
    resolved_root = resolve_root(root)
    path = resolve_config_file(resolved_root, config_file)
    settings = build_settings(resolved_root, read_config_file(path) if path else {})
    if log_level:
        settings = replace(settings, log_level=log_level.upper())
    return settings.validate()


DEFAULT_CORPUS = CorpusConfig()
DEFAULT_MODEL = ModelConfig()
DEFAULT_CHUNKING = ChunkingConfig()
DEFAULT_SCENES = SceneConfig()
DEFAULT_RETRIEVAL = RetrievalConfig()
DEFAULT_EVALUATION = EvaluationConfig()
