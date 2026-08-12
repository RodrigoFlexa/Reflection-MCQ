"""Entrypoint: python -m rmcq <comando>"""

import sys

from rmcq.cli import main

if __name__ == "__main__":
    sys.exit(main())
