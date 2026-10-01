"""The validated configuration layer (satc_rag.config): defaults, TOML overrides, and early failures."""

from dataclasses import replace
from pathlib import Path

import pytest

from satc_rag.config import (
    BGE_M3,
    ChunkingConfig,
    EvaluationConfig,
    ModelConfig,
    RetrievalConfig,
    Settings,
    load_settings,
)
from satc_rag.errors import ConfigurationError, ModelConfigurationError


def write_config(root: Path, text: str) -> Path:
    path = root / "satc-rag.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_reproduce_the_frozen_baseline(tmp_path):
    settings = Settings.defaults(tmp_path)
    assert (settings.retrieval.bm25_k1, settings.retrieval.bm25_b, settings.retrieval.bm25_epsilon) == (1.5, 0.75, 0.0)
    assert (settings.retrieval.rrf_k, settings.retrieval.hybrid_candidates) == (60, 30)
    assert settings.chunking.configs == ((256, 32), (512, 64), (1024, 128))
    assert (settings.model.repo_id, settings.model.revision, settings.model.device) == (
        BGE_M3.repo_id,
        BGE_M3.revision,
        "cpu",
    )
    assert settings.evaluation.k_values == (1, 5, 10)
    assert settings.evaluation.all_targets_k == (5, 10)
    assert settings.paths.dev_file == tmp_path / "eval" / "frozen" / "dev.json"


def test_the_pinned_identity_matches_the_cache_keys_on_disk():
    identity = BGE_M3.identity()
    assert identity == {
        "model": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "max_tokens": 8192,
        "dim": 1024,
        "normalized": True,
        "pooling": {"embedding_dimension": 1024, "pooling_mode": "cls", "include_prompt": True},
    }


def test_a_toml_file_overrides_settings_and_resolves_paths_against_the_root(tmp_path):
    write_config(
        tmp_path, 'log_level = "DEBUG"\n[retrieval]\nhybrid_candidates = 40\n[paths]\nindex_dir = "cache/index"\n'
    )
    settings = load_settings(tmp_path)
    assert settings.retrieval.hybrid_candidates == 40
    assert settings.log_level == "DEBUG"
    assert settings.paths.index_dir == tmp_path / "cache" / "index"


def test_the_config_file_can_be_named_explicitly(tmp_path):
    other = tmp_path / "other.toml"
    other.write_text("[retrieval]\nrrf_k = 10\n", encoding="utf-8")
    assert load_settings(tmp_path, other).retrieval.rrf_k == 10
    with pytest.raises(ConfigurationError, match="config file not found"):
        load_settings(tmp_path, tmp_path / "missing.toml")


@pytest.mark.parametrize(
    ("toml", "problem"),
    [
        ("[retrieval]\nbm25_k2 = 1\n", "unknown setting"),
        ("[retrieval]\nbm25_k1 = 'high'\n", "wrong type"),
        ("[chunking]\nsize = 512.5\n", "wrong type"),
        ("[scenes]\nenabled = 1\n", "wrong type"),
        ("[unknown]\nx = 1\n", "unknown config section"),
        ("[paths]\nraw = 'x.csv'\n", "unknown setting"),
        ("[retrieval\n", "cannot parse"),
    ],
)
def test_unknown_or_mistyped_settings_fail_early(tmp_path, toml, problem):
    write_config(tmp_path, toml)
    with pytest.raises(ConfigurationError, match=problem):
        load_settings(tmp_path)


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"chunking": ChunkingConfig(size=512, overlap=512, configs=((512, 512),))}, "overlap < size"),
        ({"chunking": ChunkingConfig(size=9000, overlap=64, configs=((9000, 64),))}, "would be truncated"),
        ({"chunking": ChunkingConfig(size=300, overlap=30)}, "not one of chunking.configs"),
        ({"chunking": ChunkingConfig(configs=((256, 32), (256, 16), (512, 64)))}, "same size"),
        ({"retrieval": RetrievalConfig(bm25_b=1.5)}, "bm25_b"),
        ({"retrieval": RetrievalConfig(rrf_k=0)}, "rrf_k"),
        ({"evaluation": EvaluationConfig(all_targets_k=(5, 20))}, "subset"),
        ({"evaluation": EvaluationConfig(compare_k=3)}, "compare_k"),
        ({"evaluation": EvaluationConfig(depth=5)}, "depth"),
        ({"evaluation": EvaluationConfig(k_values=(10, 5))}, "strictly increasing"),
        ({"model": ModelConfig(device="mps")}, "device"),
        ({"model": ModelConfig(batch_size=0)}, "batch_size"),
        ({"model": ModelConfig(revision="main")}, "40-character commit"),
        ({"model": ModelConfig(revision="refs/pr/130")}, "pull-request refs"),
        ({"model": ModelConfig(repo_id="BAAI/bge-small-en-v1.5")}, "not a supported pinned model"),
        ({"log_level": "LOUD"}, "log_level"),
    ],
)
def test_contradictory_or_unsupported_values_are_rejected(tmp_path, change, problem):
    with pytest.raises(ConfigurationError, match=problem):
        replace(Settings.defaults(tmp_path), **change).validate()


def test_all_problems_are_listed_together(tmp_path):
    bad = replace(Settings.defaults(tmp_path), retrieval=RetrievalConfig(bm25_k1=0, rrf_k=0), log_level="LOUD")
    with pytest.raises(ConfigurationError) as error:
        bad.validate()
    assert str(error.value).count("\n  - ") == 3


def test_an_unsupported_model_has_no_silent_substitute():
    with pytest.raises(ModelConfigurationError, match="unsupported"):
        _ = ModelConfig(repo_id="BAAI/bge-small-en-v1.5").spec


def test_the_root_must_exist(tmp_path):
    with pytest.raises(ConfigurationError, match="not a directory"):
        load_settings(tmp_path / "nowhere")


def test_the_root_can_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SATC_RAG_ROOT", str(tmp_path))
    assert load_settings().paths.root == tmp_path.resolve()


def test_settings_serialize_for_run_records(tmp_path):
    record = Settings.defaults(tmp_path).as_dict()
    assert record["paths"]["root"] == str(tmp_path)
    assert record["model"]["spec_identity"]["pooling"]["pooling_mode"] == "cls"
