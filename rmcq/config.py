"""
Registro central: modelos e runtime dos backends.

Fonte de verdade única para o que um backend precisa saber. Para colocar um
modelo novo no ar, adicione uma entrada em MODELS (ou declare-a por variável
de ambiente, como já é feito para Azure e Ollama) — nenhum outro módulo
precisa mudar.
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


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

MODELS_DIR = ROOT / "models"
HF_HOME = Path(os.environ.get("HF_HOME") or (MODELS_DIR / "hf_cache"))

# Cache em disco das respostas de backends de API (Azure, Ollama). Reaproveita
# a mesma chamada entre reruns em vez de pagar (ou esperar) de novo.
CACHE_DIR = ROOT / "cache"


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """
    Um modelo utilizável por `rmcq.backends.get_backend`.

    `provider` decide quem atende a chamada:
      - "hf"     modelo com pesos, local. O ENGINE (vLLM ou transformers) é
                 escolhido por `--backend`/RMCQ_BACKEND, não por aqui.
      - "azure"  deployment Azure OpenAI. Sem pesos, sem VRAM.
      - "ollama" modelo servido por um `ollama serve` local ou remoto.

    `repo_id` é o identificador que o provider entende: um repo do Hugging
    Face Hub para "hf", o nome do deployment para "azure", a tag do modelo
    (ex.: "llama3.1:8b") para "ollama".
    """

    key: str
    repo_id: str
    provider: str = "hf"
    trust_remote_code: bool = False
    notes: str = ""
    extra_kwargs: dict = field(default_factory=dict)

    @property
    def is_api(self) -> bool:
        """Sem download, sem VRAM neste processo: fala com um servidor."""
        return self.provider != "hf"


# Alguns modelos de exemplo, prontos para rodar via vLLM ou transformers
# (provider="hf" == pesos locais). Apague os que não usar, adicione os seus.
MODELS: dict[str, ModelSpec] = {
    "phi4-mini": ModelSpec(
        key="phi4-mini",
        repo_id="microsoft/Phi-4-mini-instruct",
        notes="MIT. 3.8B.",
    ),
    "qwen3-8b": ModelSpec(
        key="qwen3-8b",
        repo_id="Qwen/Qwen3-8B",
        notes="Apache 2.0. 8B. Pensamento híbrido — ver QWEN_ENABLE_THINKING.",
    ),
    "llama3-8b": ModelSpec(
        key="llama3-8b",
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        notes=(
            "Licença Llama 3, aprovação manual: aceite os termos em "
            "huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct e defina HF_TOKEN."
        ),
    ),
    "mistral-7b": ModelSpec(
        key="mistral-7b",
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        notes="Apache 2.0. 7B.",
    ),
}

# ---------------------------------------------------------------------------
# Professores de API: Azure OpenAI
# ---------------------------------------------------------------------------
# O nome do modelo É o nome do deployment — assim o que roda é o que está
# escrito, sem apelido genérico escondendo qual deployment respondeu de fato.
#
#     RMCQ_AZURE_DEPLOYMENTS=gpt-5-mini-petrobras,gpt-4o-petrobras

AZURE_DEPLOYMENTS = tuple(
    d.strip() for d in os.environ.get("RMCQ_AZURE_DEPLOYMENTS", "").split(",") if d.strip()
)


def _register_azure_deployments() -> None:
    for nome in AZURE_DEPLOYMENTS:
        if nome in MODELS:
            raise ValueError(
                f"RMCQ_AZURE_DEPLOYMENTS: {nome!r} colide com um modelo já registrado."
            )
        MODELS[nome] = ModelSpec(
            key=nome,
            repo_id=f"azure://{nome}",
            provider="azure",
            extra_kwargs={"deployment": nome},
            notes="Deployment Azure OpenAI declarado em RMCQ_AZURE_DEPLOYMENTS.",
        )


_register_azure_deployments()

# ---------------------------------------------------------------------------
# Modelos servidos por Ollama
# ---------------------------------------------------------------------------
# A tag do Ollama (a mesma usada em `ollama pull <tag>`) vira a chave do
# modelo, então o que está declarado aqui é exatamente o que roda.
#
#     RMCQ_OLLAMA_MODELS=llama3.1:8b,mistral:7b

OLLAMA_MODELS = tuple(
    d.strip() for d in os.environ.get("RMCQ_OLLAMA_MODELS", "").split(",") if d.strip()
)


def _register_ollama_models() -> None:
    for tag in OLLAMA_MODELS:
        if tag in MODELS:
            raise ValueError(
                f"RMCQ_OLLAMA_MODELS: {tag!r} colide com um modelo já registrado."
            )
        MODELS[tag] = ModelSpec(
            key=tag,
            repo_id=tag,
            provider="ollama",
            extra_kwargs={"tag": tag},
            notes="Modelo Ollama declarado em RMCQ_OLLAMA_MODELS.",
        )


_register_ollama_models()

ALL_MODELS = tuple(MODELS)


# ---------------------------------------------------------------------------
# Runtime: modelos locais (hf / vLLM)
# ---------------------------------------------------------------------------

BACKEND = _env_str("RMCQ_BACKEND", "vllm").lower()
TORCH_DTYPE = _env_str("RMCQ_DTYPE", "bfloat16")
LOAD_IN_4BIT = _env_str("RMCQ_LOAD_IN_4BIT", "0") in ("1", "true", "True")

SEED = _env_int("RMCQ_SEED", 42)
MAX_MODEL_LEN = _env_opt_int("RMCQ_MAX_MODEL_LEN")

HF_BATCH_SIZE = _env_int("RMCQ_HF_BATCH_SIZE", 16)

VLLM_GPU_UTIL = _env_float("RMCQ_VLLM_GPU_UTIL", 0.90)

# Modo determinístico do vLLM: com temperatura 0 a decodificação é greedy,
# mas o continuous batching torna a aritmética dependente da composição do
# lote, e num empate quase exato o argmax vira. Estas flags tiram essa fonte
# de ruído ao custo de throughput (estimar 1,5-3x mais lento). Desligue
# (RMCQ_VLLM_DETERMINISTIC=0) quando velocidade importar mais que
# reprodutibilidade byte a byte.
VLLM_DETERMINISTIC = _env_str("RMCQ_VLLM_DETERMINISTIC", "1") in ("1", "true", "True")
VLLM_MAX_NUM_SEQS = _env_opt_int("RMCQ_VLLM_MAX_NUM_SEQS") or 32

# Qwen3 é de pensamento híbrido; os outros modelos ignoram esta flag.
QWEN_ENABLE_THINKING = _env_str("RMCQ_QWEN_ENABLE_THINKING", "0") in ("1", "true", "True")

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


def hf_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    return None


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------
# Credenciais NÃO moram aqui: só os nomes das variáveis. O backend lê o
# ambiente na hora de instanciar o cliente, para que nada de secreto possa
# vazar num log de config.

# Duas formas de dizer para onde ir, mutuamente exclusivas (o SDK recusa as
# duas juntas):
#
#   AZURE_OPENAI_ENDPOINT  -> o SDK monta {endpoint}/openai/deployments/{modelo}/...
#   AZURE_OPENAI_BASE_URL  -> o SDK usa a URL COMO ESTÁ (o que gateway
#                             corporativo costuma exigir). Tem precedência
#                             quando as duas estiverem definidas.
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

AZURE_CONCURRENCY = _env_int("RMCQ_AZURE_CONCURRENCY", 4)
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

AZURE_CACHE = _env_str("RMCQ_AZURE_CACHE", "1") in ("1", "true", "True")


def azure_deployment(model_key: str) -> str:
    """Nome do deployment Azure de um modelo de API."""
    spec = MODELS[model_key]
    if spec.provider != "azure":
        raise ValueError(f"{model_key!r} não é modelo Azure (provider={spec.provider!r})")
    return spec.extra_kwargs.get("deployment") or model_key


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = _env_str("RMCQ_OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = _env_float("RMCQ_OLLAMA_TIMEOUT", 300.0)
OLLAMA_NUM_CTX = _env_opt_int("RMCQ_OLLAMA_NUM_CTX")
# Quanto tempo o servidor mantém o modelo carregado após a última chamada.
OLLAMA_KEEP_ALIVE = _env_str("RMCQ_OLLAMA_KEEP_ALIVE", "5m")
