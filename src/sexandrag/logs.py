"""Logging setup for the command line.

Library modules log through `logging.getLogger(__name__)` and never print. The CLI calls
configure_logging once: diagnostics go to stderr, command results to stdout. Third-party
libraries (transformers, huggingface_hub, sentence-transformers) stay at WARNING unless --debug.
Log messages name files, counts, ids and hashes, never large transcript passages.
"""

import logging
import sys

PACKAGE_LOGGER = "sexandrag"
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
NOISY_LIBRARIES = ("transformers", "sentence_transformers", "huggingface_hub", "urllib3", "filelock", "torch")
PLAIN_FORMAT = "%(levelname)-7s %(message)s"
DEBUG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str = "INFO", *, debug: bool = False) -> None:
    """Send the package's log records to stderr at `level` (DEBUG and timestamps with debug=True)."""
    level = "DEBUG" if debug else level.upper()
    if level not in LEVELS:
        raise ValueError(f"unknown log level {level!r}; choose one of {', '.join(LEVELS)}")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(DEBUG_FORMAT if debug else PLAIN_FORMAT, "%H:%M:%S"))
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False
    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.DEBUG if debug else logging.WARNING)
