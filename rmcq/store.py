"""
Store JSONL retomável e utilitários de log.

O motivo de existir: a grade completa leva dezenas de horas de GPU. Qualquer
execução vai ser interrompida — OOM, preempção da fila, queda de conexão. Sem
retomada por chave, cada interrupção joga fora todo o trabalho feito.

Contrato: `append` grava e dá flush a cada chamada. `done_keys` lê o que já
existe. Uma linha corrompida no fim do arquivo (efeito de kill no meio de um
write) é detectada e truncada em vez de derrubar a leitura.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from rmcq.config import LOG_LEVEL, LOGS_DIR

# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------

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


log = get_logger()


def log_to_file(tag: str) -> Path:
    """Espelha o log num arquivo, para runs longos em background."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logging.getLogger("rmcq").addHandler(handler)
    return path


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _default_key(row: dict[str, Any]) -> str:
    return str(row["uid"])


class JsonlStore:
    """
    Arquivo JSONL append-only com índice de chaves já concluídas.

    Uso:

        store = JsonlStore(path)
        pending = [i for i in items if i["uid"] not in store.done_keys()]
        ...
        store.append(records)
    """

    def __init__(
        self,
        path: str | Path,
        key_fn: Callable[[dict[str, Any]], str] = _default_key,
    ) -> None:
        self.path = Path(path)
        self.key_fn = key_fn
        self._done: set[str] | None = None
        self._fh = None

    # -- leitura ------------------------------------------------------------

    def done_keys(self, refresh: bool = False) -> set[str]:
        """Chaves já gravadas. Cacheado; `refresh=True` relê do disco."""
        if self._done is not None and not refresh:
            return self._done

        done: set[str] = set()
        if self.path.exists():
            self._repair_if_truncated()
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        done.add(self.key_fn(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        self._done = done
        return done

    def _repair_if_truncated(self) -> None:
        """
        Corta a última linha se ela não for JSON válido.

        Acontece quando o processo morre no meio de um write. Sem isso, todo
        `done_keys` seguinte perde a linha e, pior, um `read_all` estoura.
        """
        raw = self.path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            # Termina em newline: se a última linha completa for inválida ainda
            # é possível, mas isso indica corrupção real, não truncamento.
            return

        cut = raw.rfind(b"\n")
        tail = raw[cut + 1:] if cut >= 0 else raw
        try:
            json.loads(tail.decode("utf-8"))
            # Linha final íntegra, só falta o newline. Acrescenta.
            with self.path.open("ab") as fh:
                fh.write(b"\n")
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning(
                "%s: última linha incompleta (%d bytes), truncando",
                self.path.name, len(tail),
            )
            with self.path.open("wb") as fh:
                fh.write(raw[: cut + 1] if cut >= 0 else b"")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        self._repair_if_truncated()
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def __len__(self) -> int:
        return len(self.done_keys())

    def exists(self) -> bool:
        return self.path.exists()

    # -- escrita ------------------------------------------------------------

    def append(self, rows: Iterable[Any]) -> int:
        """
        Acrescenta linhas e dá flush + fsync.

        O fsync custa alguns milissegundos por lote e vale: sem ele, um crash
        de máquina perde tudo que estava no buffer do sistema, que pode ser
        meia hora de geração.
        """
        rows = list(rows)
        if not rows:
            return 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        done = self.done_keys()

        with self.path.open("a", encoding="utf-8") as fh:
            for row in rows:
                payload = row if isinstance(row, dict) else asdict(row)
                fh.write(json.dumps(payload, ensure_ascii=False))
                fh.write("\n")
                done.add(self.key_fn(payload))
            fh.flush()
            os.fsync(fh.fileno())

        return len(rows)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self._done = set()

    def __repr__(self) -> str:
        return f"JsonlStore({self.path}, {len(self)} linhas)"


def sample_key(row: dict[str, Any]) -> str:
    """Chave para etapas com múltiplas amostras por item (self-consistency)."""
    return f"{row['uid']}#{row.get('attempt', 1)}"


# ---------------------------------------------------------------------------
# Barra de progresso que funciona com ou sem tqdm
# ---------------------------------------------------------------------------


def progress(iterable: Sequence[Any], desc: str = "", total: int | None = None):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
    except ImportError:
        return iterable


class Timer:
    """Cronômetro de contexto, para registrar custo por etapa."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
