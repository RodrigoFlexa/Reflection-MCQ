#!/usr/bin/env python3
"""
DEPRECADO. Use `python -m rmcq setup-models` (ou `python -m rmcq smoke`).

Delega para o CLI. `--smoke-test` virou o subcomando `smoke`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rmcq.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--smoke-test" in args:
        args = [a for a in args if a != "--smoke-test"]
        print("aviso: use 'python -m rmcq smoke'\n", file=sys.stderr)
        sys.exit(main(["smoke", *args]))
    print("aviso: use 'python -m rmcq setup-models'\n", file=sys.stderr)
    sys.exit(main(["setup-models", *args]))
