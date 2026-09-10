# Reflection-MCQ

Experimento de transferência de reflexões em questões de múltipla escolha, com
recuperação top-1 de uma questão do treino para cada questão de avaliação.
A mesma fonte de treino pode ser recuperada por várias questões: cada modelo
responde e reflete sobre cada fonte única uma única vez.

## Próxima run: teste + RACE

O roteiro completo para copiar e colar está em [docs/RUNBOOK.md](docs/RUNBOOK.md):
GPU → GitHub → Petrobras → GitHub → GPU, com execução em segundo plano,
retomada e instalação do mesmo RACE nos dois servidores.

No servidor GPU, dentro do ambiente Python já utilizado:

```bash
python setup_server.py gpu --gpu 3
python experiment_ops.py start prepare --gpu 3
```

O instalador preserva a pilha GPU existente. `experiment_ops.py share prepare`
envia artefatos e RACE completo comprimidos ao GitHub. Na Petrobras,
`python setup_server.py petrobras` instala o RACE desse pacote sem acesso ao
Hugging Face, e adiciona somente as dependências Azure. `start teacher` e
`start finish` restauram o pacote da etapa anterior e aplicam os limites do manifesto.

As instruções abaixo também documentam o uso manual de `run_experiment.py`.
Os scripts de operação não substituem esse entrypoint.

O preparador baixa `ehovy/race`, configuração `all` (middle + high), e grava
`data/processed/race/{train,validation,test}.jsonl`. Ele resolve a revisão do
Hugging Face para um commit e salva contagens e hashes em `dataset_manifest.json`.
Os demais datasets continuam usando seus arquivos já preparados.

Os defaults da nova run são:

- datasets: `aqua,arc,logiqa2,openbookqa,race`;
- modelos: `phi2,deepseek-r1-distill-llama-8b,llama3.1-8b,phi4-mini,mistral-7b-instruct,qwen3-8b`;
- split: `test`; perfil de geração: `final`.

Os novos checkpoints são `microsoft/Phi-4-mini-instruct`,
`mistralai/Mistral-7B-Instruct-v0.3` e `Qwen/Qwen3-8B`. A presença dos pesos no
cache deve ser verificada no servidor GPU. O backend solicitado não é
substituído automaticamente se estiver indisponível.

Para um piloto em validação, use `--eval-split validation`. `--eval-cap` e
`--train-cap` existem para ensaios; a produção pesquisa o treino completo e
avalia o split completo. `--validation-cap` continua sendo um alias de `--eval-cap`.

## Execução nos dois servidores

O prepare imprime o `experiment_id`. Transporte `experiment_exchange/<id>`
pelo Git. No servidor Petrobras, após configurar o Azure no `.env`:

```bash
python -u run_experiment.py teacher --experiment-id <id>
```

Transporte a pasta `teacher/` e `teacher_receipt.json` de volta ao servidor GPU:

```bash
python -u run_experiment.py finish --experiment-id <id> --gpu 3
```

As etapas posteriores usam split, modelos, juiz, backend e perfil congelados
no manifesto. Os limites de ambiente relevantes devem corresponder aos do
prepare; divergências interrompem a etapa antes da geração.

Resultados: `data/results/reflection_top1/<id>/analysis/all_outcomes.jsonl`
e `accuracy.csv`. Avaliações individuais usam `test.jsonl` ou `validation.jsonl`
conforme o manifesto. Para consultar progresso:

```bash
python run_experiment.py status --experiment-id <id>
```

Retome sem `--fresh` para aproveitar checkpoints. Não use `--fresh` em uma
execução interrompida: essa opção reinicia os artefatos da etapa.

## Condições mantidas

Estudantes: `baseline`, `self_simple`, `self_complex`, `teacher_simple`,
`teacher_complex`. GPT-5.4 Petrobras: `baseline`, `self_simple`, `self_complex`.
Não foi acrescentada uma condição de caso recuperado sem reflexão.

## Perfil final de geração

| Modelo | Respostas de treino/teste | Reflexão simples | Reflexão complexa |
|---|---:|---:|---:|
| Phi-2 | 512 | até 768 | até 768 |
| DeepSeek-R1-Distill-Llama-8B | 4096 | 4096 | 4096 |
| Llama, Phi-4-mini, Mistral, Qwen3 | 1024 | 1024 | 1024 |
| GPT-5.4 Petrobras | teto efetivo Azure | teto efetivo Azure | teto efetivo Azure |

O teto efetivo padrão do Azure é 4000, incluindo raciocínio interno. As variáveis
`RMCQ_AZURE_MAX_TOKENS`, `RMCQ_AZURE_REASONING_MIN_TOKENS` e
`RMCQ_AZURE_REASONING_EFFORT` são registradas. O perfil final exige teto explícito.
No Phi-2, somente o orçamento de geração de reflexão se adapta ao espaço
restante da janela de 2048 tokens. O prompt completo é preservado.

O perfil final faz uma tentativa de geração, sem repetir por comprimento ou
resposta vazia. Repetições de infraestrutura do Azure continuam separadas.
Respostas usam temperatura 0; reflexões locais usam 0.7, configurável por
`--reflection-temperature`. GPT-5 não recebe temperatura. Essa configuração
não constitui uma busca pelo melhor desempenho do DeepSeek em temperatura.

Qwen3 recebe `enable_thinking=False`. Blocos think são separados da saída
utilizada em todos os backends, com texto bruto e contagens preservados nos
checkpoints do perfil final. O raciocínio privado da API não é exposto.

Saídas completas são mantidas mesmo que estejam erradas. Saídas incompletas
são guardadas para auditoria, mas não avaliadas ou transferidas como memórias.
Prompts que não cabem recebem uma marcação por item; o lote continua.
Memórias não são recortadas no perfil final.

## Gráficos e filtros

Instale `requirements-analysis.txt` e use
`validation_accuracy_by_similarity.ipynb`, agora compatível com teste e validação.
Configure `EXPERIMENT_ID` e os filtros no início:

- `EXCLUDE_FLAGS`: por exemplo `length_exhausted`, `partial_think`,
  `context_exceeded`, `content_filter`, `budget_reduced_for_context`,
  `thinking_removed`, `embedding_truncated`, `judge_fallback` ou `unresolved`;
- `EXCLUDE_METHODS`: motivos específicos em `eval_method`;
- `CONDITIONS`: condições a comparar;
- `RACE_SUBSETS`: `middle`, `high` ou ambos;
- `PAIRED=True`: mesmas questões entre as condições de cada modelo;
- `RESOLVED_ONLY=True`: somente questões resolvidas; combine com `PAIRED` para
  uma interseção resolvida nas condições comparadas;
- `PLOT_METRIC`: `accuracy_all` (acertos/selecionadas) ou `accuracy`
  (acertos/resolvidas).

O padrão inclui todas as marcações. Os quantis de similaridade são fixados no
conjunto original. Cada seleção salva figuras, métricas e filtros em uma pasta
própria, com total original, selecionado, excluído e cobertura.

Também é possível exportar uma seleção sem abrir o notebook:

```bash
python analyze_experiment.py --outcomes data/results/reflection_top1/<id>/analysis/all_outcomes.jsonl --exclude-flags length_exhausted,context_exceeded --paired --resolved-only
```

Um filtro não transforma uma saída descartada em resposta resolvida. Para
medir o efeito de usar uma reflexão incompleta, é necessária outra execução das
condições dependentes. O notebook `high_budget_reflection_experiment.ipynb`
continua sendo uma variante separada que aceita truncamentos e recorta memórias.

## RACE e contexto

O artigo completo vira `context`; cada pergunta tem UID próprio, mesmo quando
compartilha `example_id` com outras perguntas. A divisão middle/high e um hash
do artigo são preservados. Artigos do RACE presentes no split de avaliação são
excluídos dos candidatos de treino e a remoção é auditada.

A recuperação mantém `BAAI/bge-large-en-v1.5` e o texto contexto + pergunta +
alternativas, sem gabarito ou justificativa. O limite do embedder pode cortar
textos longos antes de chegar à pergunta; os pares registram contagens e
`embedding_truncated`. Essa marcação é independente do contexto do gerador.

## Notebook de análise final

Abra `test_accuracy_by_similarity.ipynb` e execute as células em ordem. Ele lê a
run de teste `f8009a56a83b` e restaura seu pacote `finish` local automaticamente
se os resultados ainda não estiverem extraídos. Não faz chamadas aos modelos.

O controle inicial `USE_BASELINE_ON_FAILURE=True` substitui respostas
experimentais indisponíveis, rejeitadas pelos filtros ou abaixo do threshold
pelo baseline da mesma questão/modelo. Se o baseline também falhar, mantém a
questão no denominador como erro. Com `False`, exclui esses casos e permite
amostras diferentes entre condições; respostas erradas válidas são preservadas.

As seções apresentam auditoria, conjunto completo, filtros de qualidade,
quantis de similaridade, thresholds e médias macro/micro. A escolha de threshold
usa apenas combinações disponíveis na validação `91ccab5e5028`; não estima
thresholds para combinações sem validação. O delta no teste usa o baseline nas
mesmas questões selecionadas. As diferenças de protocolo entre runs ficam
explicitadas no notebook.

Tabelas, decisões por questão e figuras são salvas em
`data/results/reflection_top1/<id>/analysis/test_notebook_<config>/`.
As dependências estão em `requirements-analysis.txt`.

## Nova grade de validação com professor opcional

O roteiro completo, incluindo instalação, execução local, retomada e passagem
GPU → GitHub → Petrobras → GPU, está em [docs/VALIDATION_RUNBOOK.md](docs/VALIDATION_RUNBOOK.md).

```bash
python validation_ops.py start local --gpu 3
python validation_ops.py status
```

Com duas GPUs livres, a mesma run pode ser dividida em duas partições de modelos
que rodam ao mesmo tempo, no mesmo checkout, e são unidas no fim:

```bash
python validation_ops.py start local --p1 --gpu 3   # terminal 1
python validation_ops.py start local --p2 --gpu 7   # terminal 2
python validation_ops.py merge                      # quando as duas terminarem
```

O ID da run e os resultados são idênticos aos de uma GPU só; muda só o tempo.

São nove estudantes via vLLM, cinco datasets e três condições locais. GPT-5.4
é somente professor: a etapa opcional acrescenta as duas condições externas sem
regenerar baseline/autorreflexão. O notebook `validation_accuracy_by_similarity.ipynb`
aceita tanto a etapa parcial quanto a final e compara com o baseline nas mesmas
questões. O antigo notebook está preservado com o sufixo `_legacy`.

## Histórico e verificações

O protocolo novo é v5. Ele não retoma gerações v4 misturando parâmetros. As
runs antigas continuam disponíveis para status e análise; para continuar uma
run v4, use a revisão de código correspondente. `--generation-profile legacy`
reproduz os limites anteriores em uma nova run v5, sem alterar os artefatos antigos.

Detalhes: `docs/experiment_protocol.md`; histórico: `docs/experiment_protocol_v4.md`.

```bash
python -m pytest -q -p no:cacheprovider
```
