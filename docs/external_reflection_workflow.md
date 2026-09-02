# Reflexões simple/complex via Azure — fluxo entre dois servidores

Este fluxo acrescenta duas condições externas a uma execução concluída do
experimento 08:

| Formato | Autor da memória | `depth` na análise |
|---|---|---|
| simple | próprio estudante | `simple` |
| simple | GPT-5-4 Petrobras | `external_simple` |
| complex | próprio estudante | `complex` |
| complex | GPT-5-4 Petrobras | `external_complex` |

`external_simple` usa exatamente a mesma instrução e estrutura de `simple`; o
mesmo vale para `complex`. Os tetos de geração são dimensionados por backend
para não truncar raciocínio, mas o formato comparado permanece o mesmo.

`compact` e `diagnostic` não entram nos novos outcomes, thresholds ou plots. Os
caches brutos antigos não são apagados.

## Pacote Git novo

A rodada anterior baseada apenas em `diagnostic` não é reutilizada. A versão
correta usa um diretório separado:

```text
external_reflection_exchange/<experiment_id>/gpt-5-4-petrobras/simple_complex_v2/
├── exchange_manifest.json
├── requests/<student>/<dataset>/
│   ├── external_simple.jsonl
│   └── external_complex.jsonl
├── responses/<student>/<dataset>/
│   ├── external_simple.jsonl
│   └── external_complex.jsonl
└── generation_receipt.json
```

Os arquivos são divididos para respeitar o limite de tamanho do GitHub. Cada
registro carrega ID do experimento, formato, modelo e hash do prompt.

## 0. Servidor GPU — executar a nova rodada-base

Prepare o modelo exato no servidor em que o Ollama está rodando:

```bash
ollama pull deepseek-r1:8b-llama-distill-fp16
```

No `.env`, use `RMCQ_OLLAMA_NUM_CTX=16384`, `RMCQ_OLLAMA_TIMEOUT=1800` e
`RMCQ_OLLAMA_KEEP_ALIVE=30m`. Então rode do zero a configuração atual: Phi-2,
DeepSeek-R1 Distill Llama 8B via Ollama e Llama 3.1 8B Instruct, somente com
`pool=all`.

```bash
git pull --ff-only
mkdir -p logs
tmux new -s threshold08-v3
python -u run_similarity_threshold.py --gpu 3 --fresh \
  2>&1 | tee logs/threshold08-v3.log
```

`--fresh` é usado somente nessa primeira inicialização deliberada. Se o servidor
cair, repita o comando **sem** `--fresh`; o Ollama grava checkpoint a cada oito
respostas e os backends vLLM a cada lote.

O último componente do diretório impresso ao final é o novo ID. Defina-o nos
dois servidores antes dos comandos seguintes:

```bash
export EXPERIMENT_ID=<NOVO_ID_IMPRESSO_NO_LOG>
```

Não reutilize `3f47a5b014cd`: esse ID representa modelos e pools antigos.

## 1. Servidor GPU — exportar as solicitações externas

```bash
git pull --ff-only

python run_external_reflection.py export \
  --experiment-id "$EXPERIMENT_ID" \
  --reflection-model gpt-5-4-petrobras

python run_external_reflection.py status \
  --experiment-id "$EXPERIMENT_ID" \
  --reflection-model gpt-5-4-petrobras
```

O status deve listar `external_simple` e `external_complex` para cada estudante
e dataset. Em seguida:

```bash
git add "external_reflection_exchange/$EXPERIMENT_ID/gpt-5-4-petrobras/simple_complex_v2"
git commit -m "data: export simple complex requests for $EXPERIMENT_ID"
git push
```

## 2. Servidor Petrobras — gerar os dois formatos

```bash
git pull --ff-only
python -m pip install -r requirements-azure.txt
```

O `.env` local deve conter, no mínimo:

```dotenv
RMCQ_AZURE_DEPLOYMENTS=gpt-5-4-petrobras
AZURE_OPENAI_BASE_URL=<URL_DO_GATEWAY>
AZURE_OPENAI_API_KEY=<CHAVE_LOCAL>
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_CA_BUNDLE=petrobras-ca-root.pem
RMCQ_AZURE_CONCURRENCY=4
RMCQ_AZURE_REASONING_MIN_TOKENS=4000
RMCQ_AZURE_FAIL_ON_EMPTY=1
```

Com o caminho relativo acima, o PEM fica na raiz do repositório. `.env` e
`*.pem` estão ignorados pelo Git.

Execute em `tmux`:

```bash
tmux new -s external-simple-complex

python -u run_external_reflection.py generate \
  --experiment-id "$EXPERIMENT_ID" \
  --reflection-model gpt-5-4-petrobras \
  --batch-size 128 2>&1 | tee "external-simple-complex-$EXPERIMENT_ID.log"
```

O comando salva checkpoints separadamente por formato. Após uma queda, repita
exatamente o mesmo comando, sem `--fresh`.

O gerador primeiro tenta o lote inteiro. Se o gateway bloquear um prompt, ele
isola os itens daquele lote, registra `text=""` e
`finish_reason="content_filter_skipped"` somente para os prompts reconhecidos
como violação de política e continua. Erros de chave, endpoint, certificado ou
outros problemas de configuração continuam interrompendo a execução; eles não
são mascarados como filtro de conteúdo.

Confira:

```bash
python run_external_reflection.py status \
  --experiment-id "$EXPERIMENT_ID" \
  --reflection-model gpt-5-4-petrobras
```

O status deve mencionar ambos os formatos. Prompts bloqueados pelo filtro de
conteúdo podem ficar ausentes ou vazios; o servidor GPU os excluirá e auditará.

Envie as respostas:

```bash
git add "external_reflection_exchange/$EXPERIMENT_ID/gpt-5-4-petrobras/simple_complex_v2/responses"
git add "external_reflection_exchange/$EXPERIMENT_ID/gpt-5-4-petrobras/simple_complex_v2/generation_receipt.json"
git commit -m "data: add GPT-5-4 reflections for $EXPERIMENT_ID"
git push
```

Não adicione `.env`, PEM ou logs.

## 3. Servidor GPU — avaliar os estudantes

```bash
git pull --ff-only

python run_external_reflection.py status \
  --experiment-id "$EXPERIMENT_ID" \
  --reflection-model gpt-5-4-petrobras
```

Depois, em `tmux`:

```bash
tmux new -s external-finish-v2

python -u run_external_reflection.py finish \
  --experiment-id "$EXPERIMENT_ID" \
  --reflection-model gpt-5-4-petrobras \
  --student-backend vllm \
  --gpu 3 \
  --batch-size 512 2>&1 | tee "external-finish-v2-$EXPERIMENT_ID.log"
```

O `finish` carrega cada estudante apenas uma vez e avalia sucessivamente
`external_simple` e `external_complex`. Ele:

1. mantém somente `simple`, `external_simple`, `complex` e `external_complex`;
2. ignora reflexões externas vazias/ausentes sem inserir memória vazia;
3. salva checkpoints por formato;
4. grava `analysis/external_reflection_coverage.csv` por formato e dataset;
5. recalcula curvas, thresholds, políticas holdout e transições.

Se cair, repita o comando. `--strict-external` é opcional e exige 100% de
cobertura; não o use quando houver bloqueios de conteúdo.

## 4. Plots

```bash
export RMCQ_THRESHOLD_EXPERIMENT_ID="$EXPERIMENT_ID"
jupyter lab notebooks/08_similarity_reflection_threshold_plots.ipynb
```

O notebook mostra somente as quatro condições do quadro inicial. A figura
principal é separada por modelo, tem um subplot por dataset e sobrepõe baseline,
`simple`, `complex`, `external_simple` e `external_complex`, com acurácia no eixo
y e similaridade entre questões no eixo x. As demais seções incluem:

- curvas e thresholds por formato/autor;
- efeito holdout e comparação com placebo;
- `simple` estudante versus `external_simple` GPT;
- `complex` estudante versus `external_complex` GPT;
- cobertura dos bloqueios separada por formato.

Os plots são salvos em
`data/results/similarity_threshold_v2/<NOVO_ID>/plots_accuracy_v3/`.

## Não reutilizar a rodada anterior

Não copie respostas do diretório antigo diretamente para `simple_complex_v2`.
As chaves e hashes são diferentes e o `finish` v2 rejeita um pacote no formato
anterior. A pasta antiga pode permanecer no Git apenas como registro histórico.
