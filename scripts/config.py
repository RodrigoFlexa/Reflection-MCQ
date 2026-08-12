"""
DEPRECADO. O conteúdo mudou para `rmcq/config.py`.

Este arquivo existe só para que notebooks e scripts antigos que fazem
`sys.path.insert(0, "scripts"); from config import ...` continuem funcionando.
Código novo deve importar de `rmcq.config`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rmcq.config import *  # noqa: F401,F403,E402
from rmcq.config import (  # noqa: F401,E402  — nomes não exportados por *
    DatasetSpec,
    ModelSpec,
    ensure_dirs,
    hf_token,
    n_visible_gpus,
    runtime_summary,
)
