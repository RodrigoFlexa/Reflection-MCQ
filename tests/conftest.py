"""Torna a raiz do repositório e `tools/` importáveis nos testes.

Os pontos de entrada (run_experiment, experiment_ops, validation_ops) ficam na
raiz; os utilitários de uso pontual, em `tools/`. Os testes importam os dois
pelo nome do módulo, então os dois diretórios precisam estar no caminho.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
