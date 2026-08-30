"""Log e barra de progresso compartilhados pelos backends."""

from __future__ import annotations

import logging
import sys
from typing import Any, Sequence

from rmcq.config import LOG_LEVEL

_CONFIGURED = False


def get_logger(name: str = "rmcq") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        root = logging.getLogger("rmcq")
        root.addHandler(handler)
        root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        root.propagate = False
        _CONFIGURED = True
    return logger


def progress(iterable: Sequence[Any], desc: str = "", total: int | None = None):
    """Barra de progresso se `tqdm` estiver instalado; passa direto senão."""
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
    except ImportError:
        return iterable
