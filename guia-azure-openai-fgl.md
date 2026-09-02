# Azure OpenAI — guia de implementação (chat + modelos de reasoning)

> Instruções para implementar um cliente Azure OpenAI robusto, incluindo
> suporte a modelos de *reasoning* (`gpt-5`, `o1`, `o3`, `o4-mini`). Modelo
> alvo desta implementação: **gpt-5 completo** (não o `gpt-5-mini`) — mas o
> nome do deployment deve ficar **configurável** por variável de ambiente,
> nunca fixo no código.

## Por que não basta chamar o SDK direto

Um cliente "de livro-texto" (`AzureOpenAI(...).chat.completions.create(...)`)
quebra de três formas em produção:

1. **Modelos de reasoning** (`o1`, `o3`, `o4-mini`, família `gpt-5` — incluindo
   o gpt-5 completo) usam `max_completion_tokens` em vez de `max_tokens`,
   rejeitam `temperature` customizada, e gastam parte desse orçamento em
   **raciocínio interno antes** de responder. Um orçamento pensado para uma
   resposta curta (ex.: 64 tokens) é consumido inteiro pelo raciocínio e a
   API devolve `content=""` com `finish_reason="length"` — parece bug de
   prompt, mas é comportamento normal do modelo com orçamento baixo.
2. **Gateways/deployments variam** em quais parâmetros opcionais aceitam.
   Alguns rejeitam `seed`, `response_format` etc. mesmo quando o modelo os
   suporta. Se a chamada falhar por causa disso, o parâmetro deve ser
   removido e a chamada repetida — não a execução inteira abortada.
3. **Resposta vazia não é abstenção.** Se um erro de configuração (deployment
   errado, orçamento insuficiente, filtro de conteúdo) faz o modelo devolver
   `content=""`, e isso for tratado como "o modelo não sabe", o resultado é
   uma saída inteira e plausível — e completamente sem sentido. Isso precisa
   falhar alto, não silenciosamente.

## Interface

Uma única função de entrada, para poder trocar de modelo/backend sem tocar no
resto do código:

```python
def complete(
    prompt: str, *, system: str | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str: ...
```

## Configuração

```python
provider: str = "azure"
deployment: str = "gpt-5"      # nome do DEPLOYMENT no Azure — vem de env var, nunca hardcoded
temperature: float = 0.0
max_tokens: int = 512
seed: int | None = 1234
max_retries: int = 6
backoff_base: float = 2.0
backoff_max: float = 60.0

# --- reasoning ---
api_style: str = "auto"        # auto | chat | reasoning
reasoning_min_tokens: int = 4000  # piso de tokens para reasoning; 0 = não manda cap
reasoning_effort: str = "low"     # minimal|low|medium|high|"" (omite)
send_temperature: bool = True     # alguns gateways rejeitam até em modelo chat

# --- saúde ---
fail_on_empty: bool = True
health_check_calls: int = 5
max_empty_rate: float = 0.2
```

Credenciais (nunca no mesmo lugar que a config acima — vêm só do ambiente):

```
AZURE_OPENAI_ENDPOINT=https://seu-recurso.openai.azure.com/
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-10-21

# nome do deployment — configurável, aqui apontando para gpt-5 completo
LLM_DEPLOYMENT=gpt-5
```

## Detecção de modelo de reasoning

A checagem é sobre o **nome do deployment** (não sobre o nome "oficial" do
modelo), porque deployments corporativos costumam ter sufixo/prefixo próprio:

```python
REASONING_MARKERS = ("gpt-5", "gpt5", "o1-", "o3-", "o4-", "-o1", "-o3", "-o4")

def is_reasoning_deployment(name: str) -> bool:
    n = (name or "").lower()
    return any(m in n for m in REASONING_MARKERS)
```

`is_reasoning_deployment("gpt-5")` deve dar `True` — isso é o que garante que
o gpt-5 completo (não só o mini) recebe o tratamento correto abaixo.

| | modelo chat (gpt-4o etc.) | modelo reasoning (gpt-5, o1/o3/o4-mini) |
|---|---|---|
| limite de tokens | `max_tokens` | `max_completion_tokens` |
| `temperature` | enviado | **nunca** enviado |
| `seed` | enviado | **nunca** enviado |
| `reasoning_effort` | não se aplica | enviado se configurado |
| piso de orçamento | nenhum | `max(max_tokens, reasoning_min_tokens)` |

```python
def budget(max_tokens: int, reasoning: bool, reasoning_min_tokens: int) -> int:
    if not reasoning:
        return max_tokens
    if reasoning_min_tokens <= 0:
        return 0  # omite o cap por completo
    return max(max_tokens, reasoning_min_tokens)

def build_kwargs(deployment, messages, json_mode, max_tokens, temperature,
                  reasoning, reasoning_min_tokens, reasoning_effort,
                  send_temperature, seed) -> dict:
    kwargs = {"model": deployment, "messages": messages}
    b = budget(max_tokens, reasoning, reasoning_min_tokens)
    if b > 0:
        kwargs["max_completion_tokens" if reasoning else "max_tokens"] = b
    if not reasoning and send_temperature:
        kwargs["temperature"] = temperature
    if seed is not None and not reasoning:
        kwargs["seed"] = seed
    if reasoning and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs
```

**Nota para gpt-5 completo:** ele tende a "pensar mais" que o gpt-5-mini. Se
aparecerem respostas vazias com `finish_reason="length"`, o primeiro ajuste é
subir `reasoning_min_tokens` (experimente 6000–8000) antes de suspeitar de
outra coisa.

## Parâmetro rejeitado pelo gateway: remover e repetir

```python
OPTIONAL_PARAMS = (
    "seed", "temperature", "frequency_penalty", "presence_penalty",
    "response_format", "max_tokens", "max_completion_tokens", "reasoning_effort",
)

def maybe_drop_parameter(exc: Exception, kwargs: dict, unsupported: set) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status not in (400, 422):
        return False
    message = str(exc)
    for name in OPTIONAL_PARAMS:
        if name in unsupported or name not in kwargs:
            continue
        if re.search(rf"\b{re.escape(name)}\b", message):
            unsupported.add(name)
            return True  # remova `name` de kwargs e repita a chamada
    return False
```

Memorize o parâmetro rejeitado (`unsupported`) para não reenviá-lo nas
próximas chamadas do mesmo cliente.

## Retry com backoff

Erros transitórios (`408, 409, 429, 500, 502, 503, 504`, ou nomes de exceção
contendo `RateLimit`/`Timeout`/`APIConnection`/`ServiceUnavailable`) devem ser
repetidos com backoff exponencial + jitter, respeitando o header
`Retry-After`/`Retry-After-Ms` quando presente:

```python
delay = min(backoff_max, backoff_base ** attempt) * (0.5 + random.random())
time.sleep(max(delay, retry_after_seconds(exc)))
```

## Resposta vazia = falha, não abstenção

A primeira chamada real que devolve `content=""`, ou uma taxa sustentada de
vazias depois de algumas chamadas, deve **abortar** com um erro que já
diagnostique a causa mais provável — em especial o padrão clássico de
reasoning-model esgotando o orçamento em raciocínio
(`finish_reason == "length"` com `reasoning_tokens` alto): nesse caso a
correção é subir `max_tokens`/`reasoning_min_tokens`. Sem essa guarda, um
backend mal configurado gera silenciosamente uma saída inteira e plausível,
mas vazia de conteúdo real.

## Cache (opcional, mas recomendado)

Cachear cada chamada em disco por hash de
`deployment | temperature | max_tokens | seed | json_mode | system | prompt`
torna reruns reprodutíveis e evita pagar duas vezes pelo mesmo prompt.

## Resumo executável (para colar como instrução a um agente/LLM)

> Implemente um cliente Azure OpenAI com uma função única `complete(prompt,
> system=None, json_mode=False, max_tokens=None, temperature=None) -> str`.
> Detecte se o deployment é um modelo de reasoning por substring no nome
> (`gpt-5`, `gpt5`, `o1-`, `o3-`, `o4-` e variantes com sufixo/prefixo) — isso
> inclui o gpt-5 completo, não só o mini. Para modelos de reasoning: use
> `max_completion_tokens` em vez de `max_tokens`, nunca envie `temperature`
> nem `seed`, garanta um piso mínimo de tokens (configurável, default ~4000)
> porque o modelo consome parte do orçamento em raciocínio interno antes de
> responder, e opcionalmente envie `reasoning_effort`. Se a API rejeitar um
> parâmetro opcional (erro 400/422 mencionando o nome do parâmetro), remova-o
> e repita a chamada uma vez, memorizando para não reenviá-lo depois. Aplique
> retry com backoff exponencial + jitter em erros 429/5xx/timeout, respeitando
> `Retry-After`. Trate uma resposta vazia como falha grave (nunca como
> "abstenção"): a primeira resposta vazia, ou uma taxa sustentada delas, deve
> abortar com um erro que diagnostique a causa mais provável. Deixe o nome do
> deployment inteiramente configurável por variável de ambiente — default
> para `"gpt-5"` (o modelo completo), nunca hardcoded em outro lugar do
> código.
