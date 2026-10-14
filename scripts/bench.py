"""CLI shim: py -3 scripts/bench.py  ->  veilscan bench"""

from __future__ import annotations

import sys

from veilscan.cli import app


if __name__ == "__main__":
    sys.argv = ["veilscan", "bench", *sys.argv[1:]]
    app()
