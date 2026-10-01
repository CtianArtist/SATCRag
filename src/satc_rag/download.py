"""Fetch the source corpus from Kaggle (the repository never contains it) and check its checksum.

Needs the optional `download` extra (`pip install -e ".[download]"`) and Kaggle access. The
dataset version is pinned in the configuration, and the file's SHA-256 must match the version
the eval targets were written against; a different file is removed again rather than kept.
"""

import logging
import shutil
from pathlib import Path

from satc_rag.artifacts import file_sha256
from satc_rag.config import CorpusConfig, PathsConfig
from satc_rag.errors import CorpusChecksumError, DownloadError

log = logging.getLogger(__name__)


def download_corpus(paths: PathsConfig, corpus: CorpusConfig, force: bool = False) -> Path:
    """Download the pinned dataset version into the raw-data folder and verify its checksum."""
    target = paths.raw_csv
    if target.is_file() and not force:
        if file_sha256(target) == corpus.raw_csv_sha256:
            log.info("corpus already present and verified: %s", target)
            return target
        raise CorpusChecksumError(
            f"{target} exists but is not the pinned corpus version",
            expected=corpus.raw_csv_sha256,
            actual=file_sha256(target),
            recovery="rerun with --force to replace it",
        )
    try:
        import kagglehub
    except ImportError as exc:
        raise DownloadError(
            "kagglehub is not installed", recovery='install the download extra: pip install -e ".[download]"'
        ) from exc
    log.info("downloading Kaggle dataset %s", corpus.kaggle_dataset)
    try:
        cache = Path(kagglehub.dataset_download(corpus.kaggle_dataset))
    except Exception as exc:  # kagglehub raises many types (auth, network, missing version)
        raise DownloadError(
            f"could not download {corpus.kaggle_dataset}: {exc}",
            recovery="check your Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY) and network",
        ) from exc
    source = cache / target.name
    if not source.is_file():
        raise DownloadError(
            f"the downloaded dataset has no {target.name}", actual=sorted(p.name for p in cache.iterdir())
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    digest = file_sha256(target)
    if digest != corpus.raw_csv_sha256:
        target.unlink()
        raise CorpusChecksumError(
            "the downloaded corpus does not match the pinned version (removed it again)",
            expected=corpus.raw_csv_sha256,
            actual=digest,
            recovery="the upstream dataset may have changed; check the Kaggle dataset version in the config",
        )
    log.info("corpus downloaded and verified: %s (sha256 %s...)", target, digest[:12])
    return target
