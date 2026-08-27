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
CACHE_DIR = RESULTS_DIR / "cache"

# Pacote de troca entre esta máquina e o ambiente onde vivem os professores de
# API (Azure OpenAI). Fica FORA de results/ porque, ao contrário de todo o resto
# de results/, precisa ser versionado: é o git que transporta os dados entre os
# dois ambientes. Ver rmcq/stages/exchange.py e ROTEIRO-AZURE.md.
EXCHANGE_DIR = ROOT / "exchange"

DATASET_MANIFEST = DATA_DIR / "manifest_datasets.json"
MODEL_MANIFEST = MODELS_DIR / "manifest_models.json"

HF_HOME = Path(os.environ.get("HF_HOME") or (MODELS_DIR / "hf_cache"))

_ALL_DIRS = (
    RAW_DIR, PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR, HF_HOME,
    BASELINE_DIR, REFLECTIONS_DIR, INDEX_DIR, EVAL_DIR, RETRY_DIR,
    SELFCONS_DIR, ANALYSIS_DIR, LOGS_DIR, CACHE_DIR, EXCHANGE_DIR,
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
    # De onde saem os pesos e por onde se gera. "hf" são os modelos locais, que
    # obedecem a RMCQ_BACKEND (vllm/hf/stub). "azure" são os professores de API:
    # não têm pesos para baixar, e o backend deles é fixo, não configurável —
    # ver get_backend em rmcq/backends/__init__.py.
    provider: str = "hf"

    @property
    def is_api(self) -> bool:
        """Modelo de API: sem download, sem VRAM, sem setup-models."""
        return self.provider != "hf"


MODELS: dict[str, ModelSpec] = {
    "phi4-mini": ModelSpec(
        key="phi4-mini",
        repo_id="microsoft/Phi-4-mini-instruct",
        params="3.8B",
        roles=("student", "teacher"),
        notes=(
            "MIT. O repo traz modeling_phi3.py customizado, escrito para "
            "transformers 4.4x: ele importa LossKwargs, que sumiu da 5.x. "
            "Mantemos trust_remote_code=False e usamos o Phi3ForCausalLM nativo, "
            "que já cobre partial_rotary_factor e longrope."
        ),
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

# ---------------------------------------------------------------------------
# Professores de API: o nome do modelo É o nome do deployment
# ---------------------------------------------------------------------------
# Não há chave genérica ("gpt5") apontando para um deployment configurável. A
# razão é de proveniência: a chave do modelo nomeia os diretórios de saída
# (results/reflections/{aluno}__{professor}__{depth}/), então uma chave "gpt5"
# apontando para o deployment "gpt-5-mini-petrobras" produziria resultados
# rotulados como se fossem do GPT-5 completo. Pior, como toda etapa é retomável
# por uid, trocar o deployment por trás da mesma chave preencheria as lacunas
# do MESMO arquivo com outro modelo, sem registro de qual linha veio de qual.
#
# Então os deployments são declarados por nome, e viram chaves de modelo:
#
#     RMCQ_AZURE_DEPLOYMENTS=gpt-5-mini-petrobras,gpt-4o-petrobras
#
# e os diretórios saem como phi4-mini__gpt-5-mini-petrobras__simple. O que está
# escrito é o que rodou.

AZURE_DEPLOYMENTS = tuple(
    d.strip() for d in os.environ.get("RMCQ_AZURE_DEPLOYMENTS", "").split(",") if d.strip()
)


def _register_azure_deployments() -> None:
    """Cria um ModelSpec por deployment declarado, com a chave = nome do deployment."""
    for nome in AZURE_DEPLOYMENTS:
        if nome in MODELS:
            raise ValueError(
                f"RMCQ_AZURE_DEPLOYMENTS: {nome!r} colide com um modelo já registrado."
            )
        MODELS[nome] = ModelSpec(
            key=nome,
            repo_id=f"azure://{nome}",
            params="n/d",
            # Só PROFESSOR: sem "student", STUDENTS os exclui sozinho e nenhuma
            # etapa vai pedir baseline ou eval deles.
            roles=("teacher",),
            provider="azure",
            extra_kwargs={"deployment": nome},
            notes="Deployment Azure OpenAI declarado em RMCQ_AZURE_DEPLOYMENTS.",
        )


_register_azure_deployments()

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


def _parse_role(var: str, role: str) -> tuple[str, ...]:
    """
    Restringe um dos papéis, sem mexer no outro.

    Existe para a máquina que só roda uma parte do pipeline. No ambiente dos
    professores de API, `RMCQ_TEACHERS=gpt5` faz `python -m rmcq reflect` sem
    argumento nenhum já significar "gpt5", em vez de "todos os professores
    ativos" — que ali incluiria os modelos locais e tentaria carregar pesos que
    não existem naquela máquina.
    """
    default = tuple(k for k in ACTIVE_MODELS if role in MODELS[k].roles)
    raw = os.environ.get(var, "").strip()
    if not raw or raw.lower() in ("all", "todos", "*"):
        return default

    keys = tuple(k.strip() for k in raw.split(",") if k.strip())
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise ValueError(f"{var} contém modelo(s) desconhecido(s): {unknown}. Válidos: {list(ALL_MODELS)}")
    wrong_role = [k for k in keys if role not in MODELS[k].roles]
    if wrong_role:
        raise ValueError(f"{var}: {wrong_role} não tem o papel {role!r} em MODELS.")
    inactive = [k for k in keys if k not in ACTIVE_MODELS]
    if inactive:
        raise ValueError(
            f"{var}: {inactive} não está em RMCQ_ACTIVE_MODELS "
            f"(ativos: {list(ACTIVE_MODELS)}). Acrescente lá primeiro."
        )
    return tuple(k for k in ACTIVE_MODELS if k in set(keys))


STUDENTS = _parse_role("RMCQ_STUDENTS", "student")
TEACHERS = _parse_role("RMCQ_TEACHERS", "teacher")
INACTIVE_MODELS = tuple(k for k in ALL_MODELS if k not in ACTIVE_MODELS)

# Máquina que só sabe falar com API: sem GPU, sem pesos baixados. Com isto
# ligado, pedir um modelo local falha com uma mensagem que explica o que houve,
# em vez de estourar lá dentro do transformers depois de minutos procurando
# checkpoint. É o modo do ambiente onde só a etapa `reflect` roda.
API_ONLY = _env_str("RMCQ_API_ONLY", "0") in ("1", "true", "True")


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

# Modo determinístico do vLLM. Ligado por padrão.
#
# Por quê: com temperatura 0 a decodificação é greedy e "deveria" ser
# reprodutível, mas o continuous batching torna a aritmética dependente da
# COMPOSIÇÃO do lote — a ordem das reduções em ponto flutuante muda, e num
# empate quase exato o argmax vira. Medido nesta base: 10.908 itens da grade do
# notebook 05 caíram em fallback (prompt byte a byte idêntico ao do baseline,
# conferido por hash) e mesmo assim só 94,44% reproduziram a letra do baseline.
# Isso é ~5,6% de ruído puro, que com n=300 dá um desvio-padrão de utility de
# ±0,0136 — do tamanho de praticamente todos os efeitos que a grade quer medir.
#
# O que cada flag faz: prefix caching reusa estados de KV entre prompts com
# prefixo comum (e as notas criam justamente prefixos comuns), chunked prefill
# parte o prefill em pedaços de tamanho variável, e os CUDA graphs fixam formas
# de lote. Os três mudam a ordem das somas. max_num_seqs fixo tira a última
# fonte de variação de composição de lote.
#
# Custo: perde-se throughput (estimar 1,5-3x mais lento). Vale a pena — sem
# isso, nenhuma diferença entre configurações individuais da grade é
# distinguível de ruído. Para voltar ao modo rápido: RMCQ_VLLM_DETERMINISTIC=0.
VLLM_DETERMINISTIC = _env_str("RMCQ_VLLM_DETERMINISTIC", "1") in ("1", "true", "True")
VLLM_MAX_NUM_SEQS = _env_opt_int("RMCQ_VLLM_MAX_NUM_SEQS") or 32

EMBEDDER = _env_str("RMCQ_EMBEDDER", "BAAI/bge-large-en-v1.5")
EMBED_BATCH_SIZE = _env_int("RMCQ_EMBED_BATCH_SIZE", 64)

LOG_LEVEL = _env_str("RMCQ_LOG_LEVEL", "INFO").upper()

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")


# ---------------------------------------------------------------------------
# Azure OpenAI (professores de API)
# ---------------------------------------------------------------------------
# Credenciais NÃO moram aqui: só os nomes das variáveis. O backend lê o
# ambiente na hora de instanciar o cliente, para que nada de secreto possa
# vazar num log de config ou num runtime_summary(). Ver guia-azure-openai-fgl.md.

# Duas formas de dizer para onde ir, mutuamente exclusivas (o SDK recusa as
# duas juntas), e a diferença não é cosmética:
#
#   AZURE_OPENAI_ENDPOINT  -> o SDK monta {endpoint}/openai/deployments/{modelo}/...
#   AZURE_OPENAI_BASE_URL  -> o SDK usa a URL COMO ESTÁ
#
# Gateway corporativo costuma exigir a segunda: a URL não segue o padrão
# <recurso>.openai.azure.com, e montar o caminho padrão em cima dela dá 404
# "Resource Not Found" — a mesma mensagem de nome de deployment errado.
# BASE_URL tem precedência quando as duas estiverem definidas.
AZURE_ENDPOINT_VAR = "AZURE_OPENAI_ENDPOINT"
AZURE_BASE_URL_VAR = "AZURE_OPENAI_BASE_URL"
AZURE_API_KEY_VAR = "AZURE_OPENAI_API_KEY"

# Certificado raiz da autoridade corporativa, em PEM. Sem ele, a inspeção TLS
# do proxy derruba a conexão por certificado desconhecido. Caminho relativo é
# resolvido a partir da raiz do repositório.
AZURE_CA_BUNDLE = _env_str("AZURE_OPENAI_CA_BUNDLE", "")
# 2024-10-21 é GA de outubro de 2024 e NÃO conhece a família gpt-5: com ela
# a rota do deployment não existe e o Azure responde 404 "Resource Not
# Found", sem dizer que o problema é a versão. Default numa de 2025.
AZURE_API_VERSION = _env_str("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# Quantas chamadas simultâneas. A etapa reflect manda lotes de até ~380 prompts;
# sequencial isso levaria horas por lote. 4 é conservador o bastante para não
# provocar 429 num deployment corporativo compartilhado.
AZURE_CONCURRENCY = _env_int("RMCQ_AZURE_CONCURRENCY", 4)

# Teto de tokens de RESPOSTA nos professores de API. Separado de MAX_NEW_TOKENS
# porque lá 4096 é grátis (GPU local) e aqui é dinheiro.
AZURE_MAX_TOKENS = _env_int("RMCQ_AZURE_MAX_TOKENS", 1024)

# Piso de orçamento para modelos de reasoning: eles consomem tokens pensando
# ANTES de escrever. Orçamento curto devolve content="" com
# finish_reason="length". 0 desliga o cap por completo.
AZURE_REASONING_MIN_TOKENS = _env_int("RMCQ_AZURE_REASONING_MIN_TOKENS", 4000)
AZURE_REASONING_EFFORT = _env_str("RMCQ_AZURE_REASONING_EFFORT", "low")

# Resposta vazia é falha de configuração, nunca abstenção do modelo.
AZURE_FAIL_ON_EMPTY = _env_str("RMCQ_AZURE_FAIL_ON_EMPTY", "1") in ("1", "true", "True")
AZURE_HEALTH_CHECK_CALLS = _env_int("RMCQ_AZURE_HEALTH_CHECK_CALLS", 5)
AZURE_MAX_EMPTY_RATE = _env_float("RMCQ_AZURE_MAX_EMPTY_RATE", 0.2)

AZURE_MAX_RETRIES = _env_int("RMCQ_AZURE_MAX_RETRIES", 6)
AZURE_BACKOFF_BASE = _env_float("RMCQ_AZURE_BACKOFF_BASE", 2.0)
AZURE_BACKOFF_MAX = _env_float("RMCQ_AZURE_BACKOFF_MAX", 60.0)

# Cache em disco de cada chamada. Torna rerun após queda gratuito, o que importa
# porque reflect só grava o JSONL depois que o lote inteiro volta.
AZURE_CACHE = _env_str("RMCQ_AZURE_CACHE", "1") in ("1", "true", "True")


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


def azure_deployment(model_key: str) -> str:
    """
    Nome do deployment Azure de um modelo de API.

    Para os modelos declarados em RMCQ_AZURE_DEPLOYMENTS a chave e o deployment
    são a mesma string — é justamente isso que faz o nome do diretório de saída
    dizer a verdade sobre qual modelo escreveu aquelas reflexões.
    """
    spec = MODELS[model_key]
    if not spec.is_api:
        raise ValueError(f"{model_key!r} não é modelo de API (provider={spec.provider!r})")
    return spec.extra_kwargs.get("deployment") or model_key


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
        "vllm_deterministic": VLLM_DETERMINISTIC,
        "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
        "max_model_len": MAX_MODEL_LEN,
        "embedder": EMBEDDER,
        "selfcons_n": SELFCONS_N,
        "qwen_enable_thinking": QWEN_ENABLE_THINKING,
        "hf_token_present": hf_token() is not None,
    }
