# Reflection-MCQ

Extensão do paper **"Leveraging LLM Reflection to Improve Small Language Model Agents' Capabilities"** (AGENTICS 2025) para versão de journal.

Plano experimental completo em `Caderno de Experimentos.md`.

## Início rápido

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # ajuste a CUDA
pip install -r requirements.txt

cp .env.example .env          # defina CUDA_VISIBLE_DEVICES e HF_TOKEN
python -m rmcq status         # o que está instalado e o que falta

python -m rmcq setup-data                        # ~55 MB
jupyter lab notebooks/01_formatacao_e_selecao.ipynb
python -m rmcq setup-models --check              # confere acesso e volume
python -m rmcq setup-models                      # ~55 GB
python -m rmcq smoke                             # 1 questão por modelo

python -m rmcq run-all --dry-run                 # estimativa de custo
python -m rmcq run-all --limit 20                # piloto barato
python -m rmcq run-all                           # a grade completa
```

Antes de gastar GPU, o pipeline inteiro pode ser validado sem ela:

```bash
RMCQ_BACKEND=stub python -m rmcq run-all --limit 20 --embedder hashing
```

## O `.env`

Copie de `.env.example`. Ele é lido em `rmcq/__init__.py` **antes** de qualquer import de torch, que é a única janela em que `CUDA_VISIBLE_DEVICES` ainda tem efeito.

| variável | para quê |
|---|---|
| `CUDA_VISIBLE_DEVICES` | qual GPU usar. `0`, `2`, `0,1` (tensor parallel no vLLM), ou `""` para CPU |
| `HF_TOKEN` | obrigatório: o Llama 3 é gated com aprovação manual |
| `RMCQ_ACTIVE_MODELS` | quem participa, nos dois papéis. Padrão `phi4-mini,llama3-8b`; `all` abre os quatro |
| `RMCQ_K_VALUES` | quantas reflexões no prompt. Padrão `3`; `1,3,5` abre a grade |
| `RMCQ_BACKEND` | `vllm` (grade), `hf` (debug), `stub` (sem GPU) |
| `RMCQ_EMBEDDER` | padrão `BAAI/bge-large-en-v1.5` |
| `RMCQ_MAX_NEW_TOKENS` | teto de geração; o Caderno fixa 4096 |
| `RMCQ_VLLM_GPU_UTIL` | fração da VRAM reservada pelo vLLM |

Variáveis passadas na linha de comando ganham do `.env`: `CUDA_VISIBLE_DEVICES=3 python -m rmcq eval` funciona.

## Escopo da rodada atual

Dois modelos, nos dois papéis, e um valor de k.

| | |
|---|---|
| modelos ativos | `phi4-mini`, `llama3-8b` |
| fora da rodada | `qwen3-8b`, `mistral-7b` (no registro, prontos para voltar) |
| pares | 4 — 2 de autorreflexão, 2 de reflexão externa (nas duas direções) |
| profundidades | `simple`, `complex` |
| k | 3 |
| configs de avaliação | 8 |

Custo estimado (`--dry-run`, ~1500 tok/s de vLLM): baseline 0,9 h · reflexão 0,7 h · avaliação 1,8 h. **Cerca de 3,5 h de GPU no total**, contra ~26 h da grade completa.

`python -m rmcq status` imprime o escopo ativo. Para abrir depois, mude `RMCQ_ACTIVE_MODELS` e `RMCQ_K_VALUES` no `.env` — **nada é regerado**, porque baseline, reflexões e índice são compartilhados entre valores de k, e as etapas são retomáveis por `uid`.

## Comandos

| comando | etapa |
|---|---|
| `setup-data` | baixa os 5 benchmarks para `data/raw/` |
| `setup-models` | baixa os 4 modelos para `models/hf_cache/` |
| `smoke` | carrega cada modelo e responde 1 questão conhecida |
| `baseline` | **1.** todos respondem tudo, sem reflexão |
| `index` | **3.** embeddings e vizinhos top-k por dataset |
| `reflect` | **2.** professor escreve reflexão sobre cada resposta de treino |
| `eval` | **4.** aluno responde o teste com as k reflexões mais próximas |
| `retry` | controle: feedback de erro, sem reflexão |
| `analyze` | **5.** métricas da seção 5 do Caderno |
| `status` | progresso de todas as etapas |
| `run-all` | tudo, na ordem correta |

Flags comuns: `--dry-run` (estima gerações e tokens sem carregar modelo), `--backend`, `--limit N`, `--log-file`, e filtros `--students --teachers --depths -k --datasets --splits`.

**Toda etapa é retomável.** Cada linha é gravada com flush e fsync, e rodar de novo pula o que já está no JSONL, indexado por `uid`. Interromper com Ctrl-C é seguro. Um arquivo truncado por kill no meio de um write é detectado e reparado na leitura seguinte.

## As três otimizações que importam

Com os quatro modelos e k em {1,3,5} a avaliação são ~460 mil gerações, 87% do custo. O que o framework faz para não desperdiçar:

**Baseline reaproveitado.** As respostas de treino do aluno são geradas uma vez e lidas pelas 32 combinações de reflexão. As de teste são o ponto de comparação da avaliação e o ponto de partida do retry. Sem isso seriam 56 mil gerações a mais só para reproduzir textos idênticos.

**Modelo carregado uma vez por papel.** Na etapa de reflexão o laço externo é o **professor**, porque é ele que gera: 4 cargas em vez de 32. Na avaliação o laço externo é o **aluno**: 4 cargas em vez de 96.

**Índice compartilhado.** O embedding de uma questão não depende de quem respondeu nem de quem refletiu, então os vizinhos são calculados uma vez por dataset e reusados nas 96 configurações. Os vizinhos de k=1 são prefixo dos de k=5.

## Estrutura

```
rmcq/
├── config.py            # datasets, modelos, grade, runtime — fonte de verdade única
├── common.py            # prompt congelado, extrator único, Record, Cochran
├── data.py              # leitura dos splits e resolução de seletores
├── store.py             # JsonlStore retomável, log
├── retrieval.py         # embeddings e vizinhos top-k
├── cli.py               # subcomandos e --dry-run
├── backends/            # base.py (interface), vllm_backend.py, hf.py, stub.py
└── stages/              # setup_data, setup_models, baseline, reflect,
                         # evaluate, conditions, analyze

notebooks/
├── 01_formatacao_e_selecao.ipynb   # schema unificado, dedup, Cochran
└── 02_teste_inferencia.ipynb       # teste de fumaça manual

data/{raw,processed,splits}/   results/{baseline,reflections,index,eval,retry,analysis}/
scripts/                        # shims deprecados, delegam para o CLI
```

## Dados

Treino amostrado por Cochran (95%, margem 5, p = 0,5, correção finita), estratificado pelo gabarito, semente 42. Validação e teste passam inteiros.

| Dataset | Tipo | HF repo | Treino | Validação | Teste |
|---|---|---|---|---|---|
| gsm8k | processo | `openai/gsm8k` | 366 | — | 1.319 |
| aqua | processo | `deepmind/aqua_rat` | 383 | 252 | 246 |
| logiqa2 | processo | `jeggers/logiqa2_formatted` | 372 | 1.565 | 1.565 |
| arc | conhecimento | `allenai/ai2_arc` (Challenge) | 286 | 298 | 1.171 |
| openbookqa | conhecimento | `allenai/openbookqa` (main) | 356 | 500 | 500 |

Três coisas a saber antes de comparar com a literatura:

- **GSM8K não é MCQ na origem.** Os distratores vêm dos passos intermediários anotados em `<<expr=valor>>`. Os números não são comparáveis com GSM8K aberto.
- **O treino é deduplicado contra validação e teste.** Sem isso, 40% das questões de teste do LogiQA2 têm par idêntico no treino, e a recuperação devolveria a reflexão da questão idêntica — o confundidor de memorização que a extensão existe para eliminar. Custo: 3 questões no total.
- **Itens com alternativas de texto duplicado são descartados**, porque o gabarito fica ambíguo. Explica as contagens de teste um pouco abaixo do oficial.

## Modelos

| Modelo | Params | repo_id | Rodada atual | Nota |
|---|---|---|---|---|
| Phi-4 Mini | 3.8B | `microsoft/Phi-4-mini-instruct` | **ativo** | exige `trust_remote_code` |
| Llama-3-8B-Instruct | 8B | `meta-llama/Meta-Llama-3-8B-Instruct` | **ativo** | gated, exige `HF_TOKEN` |
| Qwen3-8B | 8B | `Qwen/Qwen3-8B` | fora | pensamento híbrido, transformers ≥ 4.51 |
| Mistral-7B-Instruct | 7B | `mistralai/Mistral-7B-Instruct-v0.3` | fora | — |

`setup-models` sem argumentos baixa só os ativos (~21 GB em vez de ~55 GB). `--models all` baixa os quatro.

Cada modelo ativo atua como aluno e como professor. Aluno com temperatura 0, professor com 0.8, máximo 4096 tokens, semente 42 — tudo em `rmcq/config.py`.

## Convenções

- **Um prompt, um extrator, um formato de saída**, todos em `rmcq/common.py`. Tratamento específico de modelo vira coluna no JSONL, nunca um ramo de código.
- **Abstenção não é erro.** `is_correct` é `None` quando nenhuma letra foi extraída. `accuracy` conta abstenção como erro (é o número do paper); `accuracy_answered` a ignora. A diferença separa incapacidade de resolver de incapacidade de seguir o formato.
- **`extraction_method` é registrado por linha.** Se a aderência a `FINAL ANSWER: X` cair para algum modelo, isso é resultado a reportar, não bug a mascarar no extrator.
- **Autorreflexão é a diagonal.** Aluno == professor usa o prompt na perspectiva do aluno; fora da diagonal usa a do professor. Aparece como `condition` no JSONL: `self_reflection` ou `external_reflection`.
- **Ordem das reflexões no prompt: similaridade crescente.** A mais parecida com a questão nova fica por último, adjacente a ela, onde o modelo atende mais.
- **A questão de origem não vai no prompt** por padrão. Incluir transformaria a reflexão em exemplo few-shot de uma questão quase idêntica. Para ablação: `RMCQ_INJECT_SOURCE_QUESTION=1`.

## Interpretação dos controles

`retry` mede quanto do ganho vem só de saber que errou: em 4 alternativas, isso sobe o chute de 25% para 33%. **Ele não é diretamente comparável à etapa de avaliação** — ali o modelo revisita a mesma questão sabendo que errou, aqui ele responde uma questão nova sem feedback sobre a própria resposta. O retry controla o protocolo do paper anterior, não o da extensão.

`selfcons` (self-consistency) foi **desabilitado**: o subcomando saiu da CLI e as funções em `rmcq/stages/conditions.py` levantam `RuntimeError`. O código continua no repositório como registro do desenho; para reativar, ver `SELFCONS_ENABLED` naquele arquivo.
