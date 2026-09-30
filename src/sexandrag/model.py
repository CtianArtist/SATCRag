"""The embedding model's supply chain: pinned files, offline lookup, explicit download, verification.

* Only a pinned (repo, 40-hex commit) pair from sexandrag.config.SUPPORTED_MODELS is accepted, so
  branches, tags and pull-request refs (refs/pr/N) can never be loaded.
* Normal commands look the snapshot up in the local Hugging Face cache with local_files_only=True
  and never touch the network. `sexandrag model download` is the only step that downloads.
* Every required file must match its pinned digest (git blob SHA-1 for small files, SHA-256 for
  large LFS files). The shallow check hashes everything except the 2.3 GB weights; --deep
  hashes those too.
* modules.json and the pooling config must match the pinned module list and CLS pooling, so a
  snapshot can never silently fall back to mean pooling or drop the normalization step.
* Model code is never executed from the hub: the architecture (XLM-RoBERTa) ships with
  transformers, and loading always passes trust_remote_code=False.
"""

import json
import logging
from pathlib import Path
from typing import Any

from sexandrag.artifacts import file_sha256, git_blob_sha1
from sexandrag.config import ModelSpec
from sexandrag.errors import DownloadError, ModelConfigurationError, ModelNotAvailableError

log = logging.getLogger(__name__)
LARGE_FILE_BYTES = 100 * 1024 * 1024  # files above this size are hashed only by a deep verification


def file_digest(path: Path, expected: str) -> str:
    """The digest of `path` in the same form as `expected` (SHA-256 for 64 hex chars, git blob SHA-1 for 40)."""
    return file_sha256(path) if len(expected) == 64 else git_blob_sha1(path)


def tokenizer_only_files(spec: ModelSpec) -> list[str]:
    """The small files needed to count tokens (and check the model config), without the weights."""
    return [name for name in spec.files if not name.endswith((".bin", ".safetensors"))]


def snapshot_path(spec: ModelSpec, cache_dir: Path | None = None, files: list[str] | None = None) -> Path:
    """The local snapshot folder for the pinned revision, found without any network access."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(
            snapshot_download(
                spec.repo_id,
                revision=spec.revision,
                allow_patterns=files or list(spec.files),
                cache_dir=cache_dir,
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError as exc:
        raise ModelNotAvailableError(
            f"{spec.repo_id}@{spec.revision[:12]} is not in the local model cache",
            recovery="run `sexandrag model download` once (about 2.3 GB); normal commands never download",
        ) from exc
    missing = [name for name in files or spec.files if not (path / name).is_file()]
    if missing:
        raise ModelNotAvailableError(
            f"the local snapshot of {spec.repo_id} is incomplete: missing {', '.join(missing)}",
            recovery="run `sexandrag model download` to fetch the missing files",
        )
    return path


def download_snapshot(spec: ModelSpec, cache_dir: Path | None = None, tokenizer_only: bool = False) -> Path:
    """Download exactly the pinned files of the pinned revision (the only networked model step)."""
    from huggingface_hub import snapshot_download

    files = tokenizer_only_files(spec) if tokenizer_only else list(spec.files)
    log.info("downloading %d file(s) of %s@%s", len(files), spec.repo_id, spec.revision)
    try:
        path = Path(snapshot_download(spec.repo_id, revision=spec.revision, allow_patterns=files, cache_dir=cache_dir))
    except Exception as exc:  # network, auth or disk errors from huggingface_hub, re-raised with context
        raise DownloadError(
            f"could not download {spec.repo_id}@{spec.revision[:12]}: {exc}",
            recovery="check the network connection and disk space, then retry",
        ) from exc
    log.info("model files are in %s", path)
    return path


def digest_problems(path: Path, spec: ModelSpec, files: list[str], deep: bool) -> list[str]:
    """Required files that are missing or whose content differs from the pinned digest."""
    problems = []
    for name in files:
        file = path / name
        if not file.is_file():
            problems.append(f"missing {name}")
            continue
        if not deep and file.stat().st_size > LARGE_FILE_BYTES:
            log.debug("skipping the content hash of %s (%d bytes); use --deep", name, file.stat().st_size)
            continue
        actual = file_digest(file, spec.files[name])
        if actual != spec.files[name]:
            problems.append(f"{name} digest {actual} differs from the pinned {spec.files[name]}")
    return problems


def structure_problems(path: Path, spec: ModelSpec) -> list[str]:
    """Module list and pooling config that differ from the pinned specification."""
    problems = []
    modules_file, pooling_file = path / "modules.json", path / "1_Pooling" / "config.json"
    if modules_file.is_file():
        modules = [entry.get("type") for entry in json.loads(modules_file.read_text(encoding="utf-8"))]
        if tuple(modules) != spec.modules:
            problems.append(f"modules.json lists {modules}, expected {list(spec.modules)}")
    if pooling_file.is_file():
        pooling = json.loads(pooling_file.read_text(encoding="utf-8"))
        wanted = {f"pooling_mode_{spec.pooling['pooling_mode']}_token": True}
        others = {key: value for key, value in pooling.items() if key.startswith("pooling_mode_") and value}
        if others != wanted:
            problems.append(f"pooling config enables {sorted(others)}, expected only {sorted(wanted)}")
    return problems


def verify_snapshot(path: Path, spec: ModelSpec, deep: bool = False, files: list[str] | None = None) -> None:
    """Raise ModelConfigurationError unless the snapshot holds exactly the pinned files and structure."""
    problems = digest_problems(path, spec, files or list(spec.files), deep) + structure_problems(path, spec)
    if problems:
        raise ModelConfigurationError(
            f"the local snapshot of {spec.repo_id}@{spec.revision[:12]} does not match the pinned model",
            expected="the files and digests pinned in sexandrag.config",
            actual="; ".join(problems),
            recovery="delete the cached snapshot and run `sexandrag model download` again",
        )
    log.info("model snapshot verified%s: %s@%s", " (deep)" if deep else "", spec.repo_id, spec.revision[:12])


def check_loaded_model(model: Any, spec: ModelSpec, max_tokens: int) -> None:
    """Refuse a loaded sentence-transformers model whose modules, pooling or dimension differ from the pin."""
    types = tuple(f"{type(module).__module__.rsplit('.', 1)[0]}.{type(module).__name__}" for module in model)
    problems = []
    if len(model) != len(spec.modules) or [t.rsplit(".", 1)[-1] for t in types] != [
        m.rsplit(".", 1)[-1] for m in spec.modules
    ]:
        problems.append(f"loaded modules {list(types)} differ from {list(spec.modules)}")
    pooling = model[1].get_config_dict() if len(model) > 1 else None
    if pooling != dict(spec.pooling):
        problems.append(f"pooling {pooling} differs from {dict(spec.pooling)}")
    if model.get_embedding_dimension() != spec.dim:
        problems.append(f"embedding dimension {model.get_embedding_dimension()} differs from {spec.dim}")
    if model.max_seq_length < max_tokens:
        problems.append(f"model context {model.max_seq_length} is below the configured {max_tokens} tokens")
    if problems:
        raise ModelConfigurationError(
            f"the loaded {spec.repo_id} model does not match its pinned specification",
            actual="; ".join(problems),
            recovery="re-download the model with `sexandrag model download` and check the pinned library versions",
        )
