#!/usr/bin/env python3
"""
DEPRECADO. Use `python -m rmcq setup-data`.

Delega para o CLI, então continua funcionando. Os argumentos são os mesmos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rmcq.cli import main  # noqa: E402

if __name__ == "__main__":
    print("aviso: use 'python -m rmcq setup-data'\n", file=sys.stderr)
    sys.exit(main(["setup-data", *sys.argv[1:]]))
