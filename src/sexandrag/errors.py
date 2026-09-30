"""Focused, actionable exceptions.

Every error raised on purpose derives from SexAndRagError and can say what failed, the expected
and actual values, and how to recover. The CLI prints that message without a stack trace
(unless --debug) and exits with a nonzero code. Nothing here catches or hides failures.
"""

from typing import Any


class SexAndRagError(Exception):
    """An expected, explainable failure: bad input, a stale artifact, a missing model, and so on."""

    def __init__(self, message: str, *, expected: Any = None, actual: Any = None, recovery: str | None = None):
        super().__init__(message)
        self.message = message
        self.expected = expected
        self.actual = actual
        self.recovery = recovery

    def __str__(self) -> str:
        parts = [self.message]
        if self.expected is not None:
            parts.append(f"  expected: {self.expected}")
        if self.actual is not None:
            parts.append(f"  actual:   {self.actual}")
        if self.recovery:
            parts.append(f"  fix:      {self.recovery}")
        return "\n".join(parts)


class ConfigurationError(SexAndRagError):
    """A setting is unknown, of the wrong type, out of range, or contradicts another setting."""


class CorpusError(SexAndRagError):
    """The source corpus is missing or unreadable."""


class CorpusChecksumError(CorpusError):
    """The raw CSV is not the exact file this project (and its eval targets) was built from."""


class CorpusFormatError(CorpusError):
    """The raw CSV does not have the expected structure."""


class MetadataError(SexAndRagError):
    """Project metadata (episode titles, speaker aliases) is missing or malformed."""


class ArtifactError(SexAndRagError):
    """A derived artifact (lines, chunks, index, results) cannot be used."""


class ArtifactMissingError(ArtifactError):
    """A derived artifact has not been built yet."""


class ArtifactMismatchError(ArtifactError):
    """A derived artifact is corrupt, incomplete, or does not match its inputs."""


class StaleArtifactError(ArtifactMismatchError):
    """A derived artifact was built from inputs that have since changed and must be rebuilt."""


class ModelError(SexAndRagError):
    """The embedding model cannot be used as configured."""


class ModelNotAvailableError(ModelError):
    """The pinned model files are not in the local cache (and normal commands never download)."""


class ModelConfigurationError(ModelError):
    """The model on disk or its settings differ from the pinned specification."""


class TextTooLongError(ModelError):
    """A text is longer than the model context; refusing instead of silently truncating it."""


class EvaluationError(SexAndRagError):
    """An evaluation cannot run as requested."""


class EvaluationSchemaError(EvaluationError):
    """An eval item does not follow the benchmark schema."""


class FrozenBenchmarkError(EvaluationError):
    """The frozen benchmark files no longer match their recorded hashes or split."""


class HeldOutSetError(EvaluationError):
    """An attempt to use the held-out test set without the explicit, deliberate opt-in."""


class DownloadError(SexAndRagError):
    """Fetching the corpus or the model failed."""
