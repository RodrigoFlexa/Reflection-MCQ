"""Etapas do pipeline. Cada módulo expõe `plan()` e `run()`."""

from __future__ import annotations

__all__ = [
    "setup_data", "setup_models", "baseline", "reflect",
    "evaluate", "conditions", "analyze",
]
