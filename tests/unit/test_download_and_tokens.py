"""Corpus download (with a fake kagglehub), model download failures, and the file-based token counter."""

import hashlib
import sys
import types
from dataclasses import replace

import pytest

from satc_rag.config import BGE_M3, CorpusConfig, PathsConfig
from satc_rag.download import download_corpus
from satc_rag.errors import CorpusChecksumError, DownloadError, ModelNotAvailableError
from satc_rag.model import download_snapshot
from satc_rag.tokens import tokenizer_counter

CSV = b",Season,Episode,Speaker,Line,date_job\n0,1.0,1.0,Carrie,Hello.,\n"


def fake_kagglehub(tmp_path, content: bytes):
    """A stand-in kagglehub whose dataset_download returns a folder holding one CSV."""
    folder = tmp_path / "kaggle-cache"
    folder.mkdir()
    (folder / "SATC_all_lines.csv").write_bytes(content)
    module = types.ModuleType("kagglehub")
    module.dataset_download = lambda dataset: str(folder)  # type: ignore[attr-defined]
    return module


@pytest.fixture
def paths(tmp_path):
    return PathsConfig.under(tmp_path / "project")


def test_the_corpus_is_downloaded_and_verified(paths, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub(tmp_path, CSV))
    corpus = CorpusConfig(raw_csv_sha256=hashlib.sha256(CSV).hexdigest())
    assert download_corpus(paths, corpus) == paths.raw_csv
    assert paths.raw_csv.read_bytes() == CSV
    assert download_corpus(paths, corpus) == paths.raw_csv  # already present: verified, not downloaded again


def test_a_download_with_the_wrong_checksum_is_removed_again(paths, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub(tmp_path, CSV + b"1,1.0,1.0,Big,Changed.,\n"))
    with pytest.raises(CorpusChecksumError, match="removed it again"):
        download_corpus(paths, CorpusConfig(raw_csv_sha256=hashlib.sha256(CSV).hexdigest()))
    assert not paths.raw_csv.exists()


def test_an_existing_different_corpus_is_kept_unless_forced(paths, tmp_path, monkeypatch):
    paths.raw_csv.parent.mkdir(parents=True)
    paths.raw_csv.write_bytes(b"something else")
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub(tmp_path, CSV))
    corpus = CorpusConfig(raw_csv_sha256=hashlib.sha256(CSV).hexdigest())
    with pytest.raises(CorpusChecksumError, match="--force"):
        download_corpus(paths, corpus)
    assert download_corpus(paths, corpus, force=True) == paths.raw_csv


def test_without_the_download_extra_the_error_says_how_to_install_it(paths, monkeypatch):
    monkeypatch.setitem(sys.modules, "kagglehub", None)  # makes `import kagglehub` raise ImportError
    with pytest.raises(DownloadError, match=r"\.\[download\]"):
        download_corpus(paths, CorpusConfig())


def test_a_failed_model_download_is_reported_with_a_fix(tmp_path, monkeypatch):
    import huggingface_hub

    def fail(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    with pytest.raises(DownloadError, match="network unreachable"):
        download_snapshot(replace(BGE_M3), cache_dir=tmp_path)


def test_the_token_counter_reads_a_local_tokenizer_file_without_special_tokens(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    unknown = "[UNK]"
    vocab = {unknown: 0, "[CLS]": 1, "[SEP]": 2, "hello": 3, "there": 4}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token=unknown))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    count = tokenizer_counter(path)
    assert count("hello there friend") == 3  # [CLS]/[SEP] are not counted; the unknown word still is


def test_a_missing_tokenizer_file_says_how_to_fetch_it(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="--tokenizer-only"):
        tokenizer_counter(tmp_path / "tokenizer.json")
