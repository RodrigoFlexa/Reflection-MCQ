# Reflection-MCQ — backends

Camada de acesso a modelos de linguagem usada pelo Reflection-MCQ, isolada do
resto do pipeline experimental. Uma interface (`Backend.generate`), quatro
implementações — **vLLM**, **transformers (hf)**, **Azure OpenAI** e
**Ollama** — mais um backend `stub` para testar sem GPU nem API. Junto vêm os
três prompts congelados do notebook 07 (`rmcq/prompts.py`): baseline,
reflexão e avaliação-com-reflexão.

## Início rápido

```bash
pip install -r requirements.txt          # ou só as libs do backend que for usar, ver o arquivo
cp .env.example .env                     # ajuste CUDA_VISIBLE_DEVICES, HF_TOKEN, etc.

python example.py phi4-mini              # gera via RMCQ_BACKEND do .env (padrão: vllm)
RMCQ_BACKEND=stub python example.py phi4-mini   # testa a integração sem GPU
```

```python
import rmcq  # carrega o .env antes de qualquer import de torch — importe sempre primeiro
from rmcq.backends import get_backend
from rmcq.backends.base import GenParams

with get_backend("phi4-mini") as backend:      # kind= sobrepõe RMCQ_BACKEND: "vllm" | "hf" | "stub"
    [gen] = backend.generate(["2 + 2 = ?"], GenParams(max_new_tokens=50))
    print(gen.text)
```

## Colocando um modelo novo no ar

Tudo passa por um único registro, `MODELS` em [rmcq/config.py](rmcq/config.py). Três formas de adicionar um modelo, uma por provider:

**Hugging Face (vLLM ou transformers)** — edite `MODELS` diretamente:

```python
MODELS["meu-modelo"] = ModelSpec(
    key="meu-modelo",
    repo_id="org/nome-do-repo",     # repo do Hugging Face Hub
    trust_remote_code=False,
)
```

Depois `get_backend("meu-modelo")` baixa e serve os pesos via vLLM (ou
transformers, se `--backend hf` ou se o vLLM não importar). `provider="hf"` é
o padrão.

**Ollama** — sem editar código, só o `.env`:

```bash
ollama pull llama3.1:8b
echo 'RMCQ_OLLAMA_MODELS=llama3.1:8b' >> .env
```

A tag vira a chave do modelo. `get_backend("llama3.1:8b")` fala com o
`ollama serve` local (ou remoto, via `RMCQ_OLLAMA_BASE_URL`).

**Azure OpenAI** — mesmo mecanismo, para deployments:

```bash
echo 'RMCQ_AZURE_DEPLOYMENTS=gpt-5-mini-petrobras' >> .env
```

O nome do deployment vira a chave do modelo — sem apelido genérico escondendo
qual deployment respondeu de fato.

## Prompts

`rmcq/prompts.py` traz os três prompts congelados usados no notebook 07:

| função | o quê |
|---|---|
| `build_answer_prompt(item)` | **baseline** — responde a questão sem nenhuma reflexão |
| `build_reflection_prompt(item, previous_answer, was_correct, depth, perspective)` | **reflexão** — comenta uma resposta anterior. `depth` é `simple`\|`complex`, `perspective` é `student` (autorreflexão) \|`teacher` (reflexão externa) |
| `build_eval_prompt(item, reflections, source_questions=..., source_was_correct=...)` | **avaliação com reflexão** — injeta as k reflexões recuperadas (em similaridade crescente) antes da questão nova. Sem `reflections`, devolve `build_answer_prompt()` byte a byte |

`RMCQ_EVAL_PROMPT=v1` volta ao layout antigo (reflexões antes do
enquadramento); o padrão é `v2` (enquadramento primeiro, question por
último, neutralização de letra). Ver o cabeçalho de cada seção do arquivo
para o porquê de cada decisão.

`example_reflection.py` exercita os três prompts de ponta a ponta contra um
backend de verdade: baseline -> reflexão -> avaliação.

```bash
python example_reflection.py phi4-mini
RMCQ_BACKEND=stub python example_reflection.py phi4-mini   # sem GPU
```

## Estrutura

```
rmcq/
├── __init__.py           # carrega o .env antes de qualquer import de torch/vllm
├── config.py              # MODELS (registro central) + runtime de cada backend
├── store.py                # logger e barra de progresso
├── prompts.py               # baseline, reflexão e avaliação-com-reflexão (notebook 07)
└── backends/
    ├── base.py              # Backend (ABC), Generation, GenParams — o contrato
    ├── __init__.py          # get_backend(): resolve provider/kind para a implementação certa
    ├── hf.py                 # transformers, batching ordenado por tamanho
    ├── vllm_backend.py       # vLLM, continuous batching — use para volume
    ├── azure.py               # Azure OpenAI: retry, cache em disco, detecção de reasoning
    ├── ollama.py               # HTTP para `ollama serve`
    └── stub.py                 # determinístico, sem GPU nem rede — para testar a integração

example.py                # uso mínimo de ponta a ponta (um backend, um prompt solto)
example_reflection.py     # ciclo completo: baseline -> reflexão -> avaliação
```

## O contrato `Backend`

Toda implementação garante três coisas (ver docstring em [rmcq/backends/base.py](rmcq/backends/base.py)):

1. `generate()` devolve os resultados **na mesma ordem** dos prompts recebidos.
2. O chat template do modelo é aplicado **dentro** do backend — quem chama passa o conteúdo da mensagem, não texto pré-formatado.
3. `unload()` libera o recurso de verdade (VRAM local, ou pede ao servidor remoto para descarregar), para permitir carregar o próximo modelo no mesmo processo.

`get_backend(model_key, kind=None, **kwargs)` decide qual implementação usar:
modelos `provider="azure"` ou `provider="ollama"` sempre vão para o backend
do seu provider (não faz sentido pedir um deployment Azure via `--backend
vllm`); modelos `provider="hf"` (pesos locais) usam o engine pedido em `kind`
ou em `RMCQ_BACKEND`. `--backend stub` é a exceção deliberada — troca
qualquer modelo pelo stub, para testar a integração sem gastar GPU nem API.

## O `.env`

Copie de `.env.example`. Ele é lido em `rmcq/__init__.py` **antes** de
qualquer import de torch, que é a única janela em que `CUDA_VISIBLE_DEVICES`
ainda tem efeito — por isso todo entrypoint deve `import rmcq` primeiro.

| variável | para quê |
|---|---|
| `CUDA_VISIBLE_DEVICES` | qual GPU usar (backends hf/vLLM). `0`, `0,1` (tensor parallel), `""` para CPU |
| `HF_TOKEN` | necessário para modelos gated no Hub |
| `RMCQ_BACKEND` | `vllm` \| `hf` \| `stub`, para modelos `provider="hf"` |
| `RMCQ_AZURE_DEPLOYMENTS` | deployments Azure a registrar em `MODELS` |
| `RMCQ_OLLAMA_MODELS` | tags Ollama a registrar em `MODELS` |
| `RMCQ_OLLAMA_BASE_URL` | onde está o `ollama serve` (padrão: `http://localhost:11434`) |

Variáveis passadas na linha de comando ganham do `.env`.

## Diagnóstico

```python
from rmcq.backends import available_backends
print(available_backends())
# {'stub': 'ok (sem GPU)', 'hf': 'ok', 'vllm': 'ok (vllm 0.6.x)',
#  'azure': 'indisponível: falta AZURE_OPENAI_API_KEY no .env',
#  'ollama': 'ok (6 modelo(s) no servidor http://localhost:11434)'}
```
