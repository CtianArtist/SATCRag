"""Model supply-chain checks on a fake snapshot (satc_rag.model): digests, structure, loaded-model checks."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from satc_rag.config import BGE_M3, ModelSpec
from satc_rag.errors import ModelConfigurationError, ModelNotAvailableError
from satc_rag.model import check_loaded_model, snapshot_path, tokenizer_only_files, verify_snapshot

MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
]
CLS_POOLING = {"word_embedding_dimension": 4, "pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False}


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


@pytest.fixture
def snapshot(tmp_path) -> tuple[Path, ModelSpec]:
    """A tiny snapshot folder and a spec pinning its files (small files by git blob, weights by SHA-256)."""
    files = {
        "modules.json": json.dumps(MODULES).encode(),
        "1_Pooling/config.json": json.dumps(CLS_POOLING).encode(),
        "tokenizer.json": b'{"model": "fake"}',
        "pytorch_model.bin": b"\x00weights\x00",
    }
    digests = {}
    for name, data in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(data)
        digests[name] = hashlib.sha256(data).hexdigest() if name.endswith(".bin") else git_blob(data)
    spec = replace(
        BGE_M3,
        repo_id="fake/model",
        files=MappingProxyType(digests),
        dim=4,
        pooling=MappingProxyType({"embedding_dimension": 4, "pooling_mode": "cls", "include_prompt": True}),
    )
    return tmp_path, spec


def test_a_matching_snapshot_verifies(snapshot):
    path, spec = snapshot
    verify_snapshot(path, spec, deep=True)


def test_a_missing_file_fails_verification(snapshot):
    path, spec = snapshot
    (path / "tokenizer.json").unlink()
    with pytest.raises(ModelConfigurationError, match=r"missing tokenizer\.json"):
        verify_snapshot(path, spec)


def test_changed_file_content_fails_verification(snapshot):
    path, spec = snapshot
    (path / "tokenizer.json").write_bytes(b'{"model": "tampered"}')
    with pytest.raises(ModelConfigurationError, match="differs from the pinned"):
        verify_snapshot(path, spec)


def test_changed_weights_are_caught_by_a_deep_check(snapshot):
    path, spec = snapshot
    (path / "pytorch_model.bin").write_bytes(b"\x00other weights\x00")
    with pytest.raises(ModelConfigurationError, match=r"pytorch_model\.bin"):
        verify_snapshot(path, spec, deep=True)


def test_mean_pooling_is_never_accepted_in_place_of_cls(snapshot):
    path, spec = snapshot
    pooling = dict(CLS_POOLING, pooling_mode_cls_token=False, pooling_mode_mean_tokens=True)
    (path / "1_Pooling" / "config.json").write_text(json.dumps(pooling), encoding="utf-8")
    with pytest.raises(ModelConfigurationError, match="pooling"):
        verify_snapshot(path, spec, files=["modules.json"])


def test_a_dropped_normalize_module_is_refused(snapshot):
    path, spec = snapshot
    (path / "modules.json").write_text(json.dumps(MODULES[:2]), encoding="utf-8")
    with pytest.raises(ModelConfigurationError, match=r"modules\.json lists"):
        verify_snapshot(path, spec, files=["tokenizer.json"])


def test_tokenizer_only_downloads_skip_the_weights():
    files = tokenizer_only_files(BGE_M3)
    assert "tokenizer.json" in files
    assert "pytorch_model.bin" not in files


def test_a_model_that_is_not_cached_is_reported_without_downloading(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="satc-rag model download"):
        snapshot_path(BGE_M3, cache_dir=tmp_path)


class FakeModule:
    """Stands in for a sentence-transformers module."""

    def __init__(self, config=None):
        self.config = config

    def get_config_dict(self):
        return self.config


class FakeModel(list[Any]):
    """Stands in for a loaded SentenceTransformer: a list of modules plus a few attributes."""

    max_seq_length = 8192

    def __init__(self, modules, dim=1024):
        super().__init__(modules)
        self.dim = dim

    def get_embedding_dimension(self):
        return self.dim


def module_types():
    names = ("Transformer", "Pooling", "Normalize")
    return [type(name, (FakeModule,), {}) for name in names]


def test_a_loaded_model_matching_the_pin_passes():
    transformer, pooling, normalize = module_types()
    model = FakeModel([transformer(), pooling(dict(BGE_M3.pooling)), normalize()])
    check_loaded_model(model, BGE_M3, 8192)


@pytest.mark.parametrize(
    ("pooling_config", "dim", "problem"),
    [
        ({"embedding_dimension": 1024, "pooling_mode": "mean", "include_prompt": True}, 1024, "pooling"),
        (dict(BGE_M3.pooling), 768, "embedding dimension"),
    ],
)
def test_a_loaded_model_that_differs_from_the_pin_is_refused(pooling_config, dim, problem):
    transformer, pooling, normalize = module_types()
    model = FakeModel([transformer(), pooling(pooling_config), normalize()], dim=dim)
    with pytest.raises(ModelConfigurationError, match=problem):
        check_loaded_model(model, BGE_M3, 8192)


def test_a_context_the_model_cannot_hold_is_refused():
    transformer, pooling, normalize = module_types()
    model = FakeModel([transformer(), pooling(dict(BGE_M3.pooling)), normalize()])
    with pytest.raises(ModelConfigurationError, match="below the configured"):
        check_loaded_model(model, BGE_M3, 100_000)
