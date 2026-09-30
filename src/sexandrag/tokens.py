"""Token counters used to size chunks.

Production chunk sizes are counted with the embedding model's own tokenizer, read from the
verified local model snapshot (never downloaded here), so a 512-token chunk is 512 tokens as the
embedder sees them. `regex_counter` is a dependency-free approximation for tests and fixtures.
"""

import re
from collections.abc import Callable
from pathlib import Path

from sexandrag.errors import ModelNotAvailableError

TokenCounter = Callable[[str], int]
REGEX_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def regex_counter(text: str) -> int:
    """Approximate token count: words plus punctuation marks."""
    return len(REGEX_TOKEN_RE.findall(text))


def tokenizer_counter(tokenizer_json: Path) -> TokenCounter:
    """A counter backed by a Hugging Face tokenizer file (no special tokens, no truncation)."""
    if not tokenizer_json.is_file():
        raise ModelNotAvailableError(
            f"tokenizer file not found: {tokenizer_json}", recovery="run `sexandrag model download --tokenizer-only`"
        )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    tokenizer.no_truncation()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count
