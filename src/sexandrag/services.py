"""How commands obtain the embedding model and the chunk tokenizer.

Production uses the pinned BGE-M3 snapshot from the local cache (never downloading) and loads
it once per process. Tests pass a different implementation (a deterministic fake embedder and
a regex token counter) so the whole pipeline, CLI included, runs without the corpus or the model.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from sexandrag.config import Settings
from sexandrag.embedders import Embedder, SentenceTransformerEmbedder
from sexandrag.model import snapshot_path, tokenizer_only_files, verify_snapshot
from sexandrag.tokens import TokenCounter, tokenizer_counter

log = logging.getLogger(__name__)


class Services(Protocol):
    """The model-dependent pieces a command needs."""

    def embedder(self, settings: Settings) -> Embedder:
        """The embedding model (loaded at most once per process)."""

    def embedder_identity(self, settings: Settings) -> dict[str, Any]:
        """The embedder's cache-key identity, without loading the model."""

    def tokenizer_id(self, settings: Settings) -> tuple[str, str | None]:
        """(name, revision) of the tokenizer that counts chunk sizes."""

    def token_counter(self, settings: Settings) -> TokenCounter:
        """The chunk-size token counter."""


@dataclass
class PinnedModelServices:
    """The real services: the pinned model and its tokenizer, from the verified local snapshot."""

    _embedder: Embedder | None = field(default=None, repr=False)

    def embedder(self, settings: Settings) -> Embedder:
        """Load the pinned model once; later calls reuse it (and its query-vector cache)."""
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(settings.model, settings.paths.model_cache_dir)
        return self._embedder

    def embedder_identity(self, settings: Settings) -> dict[str, Any]:
        """The pinned model's identity, from its specification (no loading)."""
        return settings.model.spec.identity()

    def tokenizer_id(self, settings: Settings) -> tuple[str, str | None]:
        """Chunk sizes are counted with the embedding model's own tokenizer."""
        spec = settings.model.spec
        return spec.repo_id, spec.revision

    def token_counter(self, settings: Settings) -> TokenCounter:
        """The model's tokenizer from the verified local snapshot (tokenizer files only)."""
        spec = settings.model.spec
        files = tokenizer_only_files(spec)
        path = snapshot_path(spec, settings.paths.model_cache_dir, files)
        verify_snapshot(path, spec, files=files)
        log.info("counting chunk tokens with the %s tokenizer from %s", spec.repo_id, path)
        return tokenizer_counter(path / spec.tokenizer_file)
