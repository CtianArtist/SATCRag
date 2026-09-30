"""Token counters used to size chunks.

The default counts tokens with the embedding model's own tokenizer, so a 512-token chunk is
512 tokens as the embedder sees them. 'regex' is a dependency-free approximation for tests.
"""
import re
from typing import Callable

TokenCounter = Callable[[str], int]
REGEX_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def regex_counter(text: str) -> int:
    """Approximate token count: words plus punctuation marks."""
    return len(REGEX_TOKEN_RE.findall(text))


def hf_counter(model_name: str, revision: str | None = None) -> TokenCounter:
    """Return a counter backed by a Hugging Face tokenizer (no special tokens, no truncation)."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_pretrained(model_name, revision=revision or "main")
    tokenizer.no_truncation()
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def get_token_counter(name: str, revision: str | None = None) -> TokenCounter:
    """Return the counter for config.TOKENIZER: 'regex' or a Hugging Face model name."""
    return regex_counter if name == "regex" else hf_counter(name, revision)
