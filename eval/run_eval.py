"""Backwards-compatible entry point: `python -m eval.run_eval [options]` is `sexandrag eval [options]`.

With no options it evaluates the DEVELOPMENT set only. The held-out test set still needs the
explicit `--split test --allow-heldout`. Run it from the repository root after `pip install -e .`.
"""

import sys

from sexandrag.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["eval", *sys.argv[1:]]))
