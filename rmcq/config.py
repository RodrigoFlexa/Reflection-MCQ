"""
Registro central: caminhos, datasets, modelos, grade experimental e runtime.

Fonte de verdade única. Nenhum outro módulo deve hardcodear repo_id, caminho,
hiperparâmetro de decodificação ou membro da grade. Ver "Caderno de
Experimentos.md", seções 2 e 3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from rmcq import ROOT  # importar rmcq primeiro garante que o .env já foi lido

# ---------------------------------------------------------------------------
# Helpers de leitura do ambiente
# ---------------------------------------------------------------------------


def _env_str(key: str, default: str) -> str:
    val = os.environ.get(key)
    return default if val is None or val == "" else val


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"{key}={val!r} não é inteiro") from None


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"{key}={val!r} não é número") from None


def _env_opt_int(key: str) -> int | None:
    val = os.environ.get(key)
    if val is None or val == "" or val.lower() == "none":
        return None
    return int(val)


def _parse_int_list(key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return tuple(sorted({int(x) for x in raw.replace(",", " ").split()}))
    except ValueError:
        raise ValueError(f"{key}={raw!r} não é uma lista de inteiros") from None


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
NOTEBOOKS_DIR = ROOT / "notebooks"

# Subdiretórios de resultados, um por etapa do pipeline.
BASELINE_DIR = RESULTS_DIR / "baseline"
REFLECTIONS_DIR = RESULTS_DIR / "reflections"
INDEX_DIR = RESULTS_DIR / "index"
EVAL_DIR = RESULTS_DIR / "eval"
RETRY_DIR = RESULTS_DIR / "retry"
SELFCONS_DIR = RESULTS_DIR / "selfcons"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
LOGS_DIR = RESULTS_DIR / "logs"

DATASET_MANIFEST = DATA_DIR / "manifest_datasets.json"
MODEL_MANIFEST = MODELS_DIR / "manifest_models.json"

HF_HOME = Path(os.environ.get("HF_HOME") or (MODELS_DIR / "hf_cache"))

_ALL_DIRS = (
    RAW_DIR, PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR, HF_HOME,
    BASELINE_DIR, REFLECTIONS_DIR, INDEX_DIR, EVAL_DIR, RETRY_DIR,
    SELFCONS_DIR, ANALYSIS_DIR, LOGS_DIR,
)


def ensure_dirs() -> None:
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo_id: str
    config: str | None
    problem_type: str              # "process" (raciocínio) ou "knowledge" (fato)
    description: str
    splits: dict[str, str]
    expected_rows: dict[str, int]  # verificado no Hub em 2026-08-11
    native_mcq: bool = True
    revision: str | None = None
    notes: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(
        key="gsm8k",
        repo_id="openai/gsm8k",
        config="main",
        problem_type="process",
        description="Aritmética escolar multi-passo, resposta numérica aberta.",
        splits={"train": "train", "test": "test"},
        expected_rows={"train": 7473, "test": 1319},
        native_mcq=False,
        notes=(
            "NÃO é múltipla escolha na origem. A conversão para MCQ usa os "
            "resultados intermediários anotados em <<expr=valor>> como "
            "distratores. Ver notebook 01."
        ),
    ),
    "aqua": DatasetSpec(
        key="aqua",
        repo_id="deepmind/aqua_rat",
        config="raw",
        problem_type="process",
        description="Álgebra e raciocínio quantitativo, 5 alternativas.",
        splits={"train": "train", "validation": "validation", "test": "test"},
        expected_rows={"train": 97467, "validation": 254, "test": 254},
        notes="Opções vêm como ['A)5', 'B)10', ...]; a letra correta está em 'correct'.",
    ),
    "logiqa2": DatasetSpec(
        key="logiqa2",
        repo_id="jeggers/logiqa2_formatted",
        config=None,
        problem_type="process",
        description="Dedução lógica (LogiQA 2.0), premissa + pergunta, 4 alternativas.",
        splits={"train": "train", "validation": "validation", "test": "test"},
        expected_rows={"train": 12567, "validation": 1569, "test": 1572},
        notes=(
            "Mirror em parquet, verificado fiel ao release oficial "
            "(datatune/LogiQA2.0, MRC/*.txt). A premissa está no campo 'text'."
        ),
    ),
    "arc": DatasetSpec(
        key="arc",
        repo_id="allenai/ai2_arc",
        config="ARC-Challenge",
        problem_type="knowledge",
        description="Ciências escolares, partição Challenge (herdado do AGENTICS).",
        splits={"train": "train", "validation": "validation", "test": "test"},
        expected_rows={"train": 1119, "validation": 299, "test": 1172},
        notes="Alguns itens usam rótulos numéricos ('1'..'4') em vez de letras.",
    ),
    "openbookqa": DatasetSpec(
        key="openbookqa",
        repo_id="allenai/openbookqa",
        config="main",
        problem_type="knowledge",
        description="Ciências elementares com fato de apoio, 4 alternativas.",
        splits={"train": "train", "validation": "validation", "test": "test"},
        expected_rows={"train": 4957, "validation": 500, "test": 500},
        notes="A pergunta está em 'question_stem'.",
    ),
}

ALL_DATASETS = tuple(DATASETS)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    params: str
    roles: tuple[str, ...]
    trust_remote_code: bool = False
    gated: bool = False
    revision: str | None = None
    ignore_patterns: tuple[str, ...] = (
        "original/*", "consolidated*", "*.pth", "*.gguf", "*.msgpack", "*.h5",
    )
    notes: str = ""
    extra_kwargs: dict = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    "phi4-mini": ModelSpec(
        key="phi4-mini",
        repo_id="microsoft/Phi-4-mini-instruct",
        params="3.8B",
        roles=("student", "teacher"),
        trust_remote_code=True,
        notes="MIT. Precisa de trust_remote_code (modeling_phi3.py customizado).",
    ),
    "qwen3-8b": ModelSpec(
        key="qwen3-8b",
        repo_id="Qwen/Qwen3-8B",
        params="8B",
        roles=("student", "teacher"),
        notes=(
            "Apache 2.0. Pensamento híbrido: o chat template aceita "
            "enable_thinking. Mantemos False para que a cadeia de raciocínio "
            "apareça na resposta visível, como nos outros modelos. "
            "Exige transformers>=4.51."
        ),
    ),
    "llama3-8b": ModelSpec(
        key="llama3-8b",
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        params="8B",
        roles=("student", "teacher"),
        gated=True,
        notes=(
            "Licença Llama 3, aprovação manual: aceite os termos em "
            "huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct e defina HF_TOKEN."
        ),
    ),
    "mistral-7b": ModelSpec(
        key="mistral-7b",
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        params="7B",
        roles=("student", "teacher"),
        notes="Apache 2.0. O repo traz consolidated.safetensors (14 GB), ignorado.",
    ),
}

ALL_MODELS = tuple(MODELS)

# ---------------------------------------------------------------------------
# Subconjunto ativo
# ---------------------------------------------------------------------------
# MODELS é o registro completo e não muda. ACTIVE_MODELS é quem participa da
# rodada atual, nos DOIS papéis. Manter os quatro no registro e filtrar aqui
# preserva repo_id, licença e notas dos que estão fora sem precisar reescrever
# nada quando eles voltarem.
#
# Rodada atual: Phi-4 Mini e Llama-3-8B. São 4 pares, e o desenho mínimo já
# contém as duas condições que interessam — a diagonal é autorreflexão, e fora
# dela ficam as duas direções da reflexão externa (o menor ensinando o maior e
# o maior ensinando o menor).
DEFAULT_ACTIVE_MODELS = ("phi4-mini", "llama3-8b")


def _parse_active() -> tuple[str, ...]:
    raw = os.environ.get("RMCQ_ACTIVE_MODELS", "").strip()
    if not raw:
        return DEFAULT_ACTIVE_MODELS
    if raw.lower() in ("all", "todos", "*"):
        return ALL_MODELS
    keys = tuple(k.strip() for k in raw.split(",") if k.strip())
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise ValueError(
            f"RMCQ_ACTIVE_MODELS contém modelo(s) desconhecido(s): {unknown}. "
            f"Válidos: {list(ALL_MODELS)}"
        )
    return keys


ACTIVE_MODELS = _parse_active()

STUDENTS = tuple(k for k in ACTIVE_MODELS if "student" in MODELS[k].roles)
TEACHERS = tuple(k for k in ACTIVE_MODELS if "teacher" in MODELS[k].roles)
INACTIVE_MODELS = tuple(k for k in ALL_MODELS if k not in ACTIVE_MODELS)


# ---------------------------------------------------------------------------
# Grade experimental (Caderno, seção 3, "Resumo variáveis do experimento")
# ---------------------------------------------------------------------------

DEPTHS = ("simple", "complex")

# Candidatos registrados no Caderno, seção 3.
K_CANDIDATES = (1, 3, 5)

# k ativo. Um valor só nesta rodada; a grade abre depois sem regerar nada, porque
# baseline, reflexões e índice são compartilhados entre valores de k.
#
# Escolhido k = 3, e não 1 nem 5, por dosagem: com k = 1 um resultado nulo é
# ambíguo (não dá para separar "reflexão recuperada não transfere" de "uma
# reflexão sobre outra questão é sinal insuficiente"), e com k = 5 o prompt
# cresce e a diluição entra como confundidor antes de sabermos se o efeito
# existe. Cada linha guarda top1_similarity e mean_similarity, então a figura de
# transferability continua possível — com a ressalva de que a atribuição limpa
# entre similaridade e utility exige k = 1, que fica para depois.
K_VALUES = _parse_int_list("RMCQ_K_VALUES", (3,))

# Condições nomeadas, usadas na coluna `condition` do JSONL.
COND_BASELINE = "no_reflection"
COND_SELF = "self_reflection"
COND_EXTERNAL = "external_reflection"
COND_RETRY = "retry_feedback"
COND_SELFCONS = "self_consistency"


def pairs(students=None, teachers=None) -> list[tuple[str, str]]:
    """
    Todos os pares (aluno, professor) viáveis.

    A diagonal (aluno == professor) é autorreflexão e usa o prompt na
    perspectiva do aluno; fora da diagonal é reflexão externa e usa o prompt na
    perspectiva do professor. É a mesma distinção da seção 4 do Caderno.
    """
    ss = tuple(students or STUDENTS)
    tt = tuple(teachers or TEACHERS)
    return [(s, t) for s in ss for t in tt]


def condition_for(student: str, teacher: str) -> str:
    return COND_SELF if student == teacher else COND_EXTERNAL


def perspective_for(student: str, teacher: str) -> str:
    return "student" if student == teacher else "teacher"


def config_tag(student: str, teacher: str, depth: str, k: int | None = None) -> str:
    """Nome de diretório de uma configuração. Estável e ordenável."""
    tag = f"{student}__{teacher}__{depth}"
    return f"{tag}__k{k}" if k is not None else tag


# ---------------------------------------------------------------------------
# Geração (Caderno, seção 2, "Notas")
# ---------------------------------------------------------------------------

SEED = _env_int("RMCQ_SEED", 42)
MAX_NEW_TOKENS = _env_int("RMCQ_MAX_NEW_TOKENS", 4096)

# Aluno: greedy determinístico.
STUDENT_GEN = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": MAX_NEW_TOKENS,
}

# Professor: amostragem a 0.8 para produzir reflexões variadas.
TEACHER_GEN = {
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": MAX_NEW_TOKENS,
}

# Self-consistency: amostragem, N amostras, voto majoritário.
SELFCONS_N = _env_int("RMCQ_SELFCONS_N", 5)
SELFCONS_GEN = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_new_tokens": MAX_NEW_TOKENS,
}

QWEN_ENABLE_THINKING = False


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

BACKEND = _env_str("RMCQ_BACKEND", "vllm").lower()
TORCH_DTYPE = _env_str("RMCQ_DTYPE", "bfloat16")
DEVICE_MAP = _env_str("RMCQ_DEVICE_MAP", "auto")
LOAD_IN_4BIT = _env_str("RMCQ_LOAD_IN_4BIT", "0") in ("1", "true", "True")

HF_BATCH_SIZE = _env_int("RMCQ_HF_BATCH_SIZE", 16)
VLLM_GPU_UTIL = _env_float("RMCQ_VLLM_GPU_UTIL", 0.90)
MAX_MODEL_LEN = _env_opt_int("RMCQ_MAX_MODEL_LEN")

EMBEDDER = _env_str("RMCQ_EMBEDDER", "BAAI/bge-large-en-v1.5")
EMBED_BATCH_SIZE = _env_int("RMCQ_EMBED_BATCH_SIZE", 64)

LOG_LEVEL = _env_str("RMCQ_LOG_LEVEL", "INFO").upper()

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")


def n_visible_gpus() -> int:
    """Quantas GPUs o .env expôs, sem importar torch."""
    val = CUDA_VISIBLE_DEVICES
    if val is None:
        return -1  # desconhecido: torch decide
    if val.strip() == "":
        return 0
    return len([x for x in val.split(",") if x.strip() != ""])


# ---------------------------------------------------------------------------
# Seleção de amostras: Cochran
# ---------------------------------------------------------------------------

COCHRAN_CONFIDENCE = 0.95
COCHRAN_MARGIN = 0.05
COCHRAN_PROPORTION = 0.50
COCHRAN_FINITE_CORRECTION = True
COCHRAN_STRATIFY_BY = "answerKey"


# ---------------------------------------------------------------------------
# Schema MCQ unificado
# ---------------------------------------------------------------------------

MCQ_FIELDS = (
    "uid", "dataset", "split", "problem_type", "context", "question",
    "choices", "answerKey", "num_choices", "rationale", "source_id",
)

CHOICE_LABELS = tuple("ABCDEFGH")


def hf_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    return None


def runtime_summary() -> dict[str, object]:
    """Snapshot do runtime, gravado junto de cada execução para reprodutibilidade."""
    return {
        "active_models": list(ACTIVE_MODELS),
        "inactive_models": list(INACTIVE_MODELS),
        "students": list(STUDENTS),
        "teachers": list(TEACHERS),
        "k_values": list(K_VALUES),
        "depths": list(DEPTHS),
        "n_pairs": len(STUDENTS) * len(TEACHERS),
        "n_eval_configs": len(STUDENTS) * len(TEACHERS) * len(DEPTHS) * len(K_VALUES),
        "backend": BACKEND,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "n_visible_gpus": n_visible_gpus(),
        "dtype": TORCH_DTYPE,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "hf_batch_size": HF_BATCH_SIZE,
        "vllm_gpu_util": VLLM_GPU_UTIL,
        "max_model_len": MAX_MODEL_LEN,
        "embedder": EMBEDDER,
        "selfcons_n": SELFCONS_N,
        "qwen_enable_thinking": QWEN_ENABLE_THINKING,
        "hf_token_present": hf_token() is not None,
    }
