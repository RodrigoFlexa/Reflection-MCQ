# External reflection via Azure — fluxo entre dois servidores

Este fluxo acrescenta `external_reflection` a uma execução **já concluída** do
experimento 08. O modelo pequeno continua sendo o estudante que responde às
questões; `gpt-5-4-petrobras` apenas escreve a memória de reflexão.

O prompt externo é idêntico ao prompt `diagnostic`. Isso permite interpretar a
comparação `diagnostic` × `external_reflection` principalmente como uma mudança
do autor da reflexão: estudante pequeno versus GPT externo.

## O que passa pelo GitHub

`data/` continua local e ignorado pelo Git. O transporte usa somente:

```text
external_reflection_exchange/
└── <experiment_id>/
    └── gpt-5-4-petrobras/
        ├── exchange_manifest.json
        ├── requests/
        │   ├── phi4-mini/<dataset>.jsonl
        │   └── mistral-7b/<dataset>.jsonl
        ├── responses/
        │   ├── phi4-mini/<dataset>.jsonl
        │   └── mistral-7b/<dataset>.jsonl
        └── generation_receipt.json
```

Os arquivos são divididos também por dataset para permanecerem abaixo do limite
de tamanho do GitHub. Eles possuem `experiment_id`, modelo, hashes e contagens.
O estágio final recusa respostas ausentes, alteradas ou pertencentes a outra
execução. O `.env` e a chave Azure nunca entram nesse diretório.

## 0. Pré-condições

- O servidor de GPU possui a execução-base completa em
  `data/results/similarity_threshold_v2/<experiment_id>`.
- Os dois servidores usam o mesmo commit do código.
- A árvore de trabalho está limpa antes de cada `git pull`.
- Substitua `<ID>` nos comandos pelo ID exibido no log do experimento 08.

Para conferir o ID no servidor de GPU:

```bash
ls -1 data/results/similarity_threshold_v2/*/manifest.json
```

## 1. Servidor de GPU — exportar os pedidos

Atualize o código e crie o pacote Git sem copiar os resultados completos:

```bash
git pull --ff-only
python run_external_reflection.py export \
  --experiment-id <ID> \
  --reflection-model gpt-5-4-petrobras

python run_external_reflection.py status \
  --experiment-id <ID> \
  --reflection-model gpt-5-4-petrobras

git add external_reflection_exchange/<ID>/gpt-5-4-petrobras
git commit -m "data: export external reflection requests for <ID>"
git push
```

O export usa as respostas das fontes já produzidas por cada estudante. Por
isso, há um arquivo separado para Phi e Mistral, mesmo que o autor externo seja
o mesmo GPT.

## 2. Servidor Petrobras — gerar somente as reflexões

Instale a dependência leve do Azure e configure um `.env` local:

```bash
git pull --ff-only
python -m pip install -r requirements-azure.txt
cp .env.example .env
```

Preencha no `.env` apenas os valores reais fornecidos no ambiente Petrobras:

```dotenv
RMCQ_AZURE_DEPLOYMENTS=gpt-5-4-petrobras
AZURE_OPENAI_BASE_URL=<URL_DO_GATEWAY>
# ou AZURE_OPENAI_ENDPOINT=<ENDPOINT_DO_RECURSO>
AZURE_OPENAI_API_KEY=<SEGREDO_LOCAL>
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_CA_BUNDLE=petrobras-ca-root.pem
RMCQ_AZURE_CONCURRENCY=4
RMCQ_AZURE_REASONING_MIN_TOKENS=4000
RMCQ_AZURE_FAIL_ON_EMPTY=1
```

Com esse valor relativo, `petrobras-ca-root.pem` deve estar na raiz clonada do
repositório, ao lado de `run_external_reflection.py`. Também é possível informar
um caminho absoluto. O backend verifica se o arquivo existe e o entrega ao
cliente HTTP como CA para a conexão TLS; se o caminho estiver errado ou o PEM
for inválido, a execução para com uma mensagem explícita antes das gerações.

Nunca execute `git add .env`. Gere as reflexões; o comando salva checkpoints a
cada lote e pode ser repetido após uma queda:

```bash
python -u run_external_reflection.py generate \
  --experiment-id <ID> \
  --reflection-model gpt-5-4-petrobras \
  --batch-size 128 2>&1 | tee external-reflection-<ID>.log

python run_external_reflection.py status \
  --experiment-id <ID> \
  --reflection-model gpt-5-4-petrobras
```

No fluxo normal, `generation_receipt.json` indica `"complete": true`. Se o
filtro de conteúdo do gateway obrigar a pular alguns prompts, respostas ausentes
ou com `text` vazio também podem ser versionadas: o estágio `finish` as excluirá
somente da condição externa e produzirá uma auditoria de cobertura.

```bash
git add external_reflection_exchange/<ID>/gpt-5-4-petrobras/responses
git add external_reflection_exchange/<ID>/gpt-5-4-petrobras/generation_receipt.json
git commit -m "data: add gpt-5-4-petrobras reflections for <ID>"
git push
```

## 3. Servidor de GPU — avaliar e recalcular os thresholds

O servidor de GPU já deve possuir a execução-base original sob o mesmo `<ID>`:

```bash
git pull --ff-only

python run_external_reflection.py status \
  --experiment-id <ID> \
  --reflection-model gpt-5-4-petrobras

tmux new -s external-reflection
python -u run_external_reflection.py finish \
  --experiment-id <ID> \
  --reflection-model gpt-5-4-petrobras \
  --student-backend vllm \
  --gpu 3 \
  --batch-size 512 2>&1 | tee external-finish-<ID>.log
```

O `finish`:

1. valida experimento e hashes do pacote recebido;
2. importa apenas reflexões com texto não vazio e hash correto;
3. executa as questões com uma memória externa por prompt;
4. salva checkpoints após cada lote;
5. acrescenta `depth=external_reflection` aos outcomes;
6. salva `analysis/external_reflection_coverage.csv` com ausentes/vazias por
   estudante e dataset;
7. recalcula curvas, thresholds, políticas holdout e transições para os cinco casos.

Esse comportamento tolerante é o padrão. Para exigir 100% das reflexões e
interromper diante de qualquer ausência, use `--strict-external` no `finish`.

Se o servidor cair, repita exatamente o mesmo comando, sem `--fresh`.

## 4. Gerar os plots

Abra e execute:

```text
notebooks/08_similarity_reflection_threshold_plots.ipynb
```

O notebook detecta `manifest["analysis_depths"]` e inclui automaticamente
`external_reflection`. Além dos painéis gerais, ele produz o gráfico direto
`diagnostic` versus `external_reflection` usando o mesmo prompt e um heatmap de
cobertura externa. Cobertura inferior a 95% aparece como alerta de possível
viés de seleção causado pelo filtro de conteúdo.

Para fixar uma execução quando houver vários IDs:

```bash
export RMCQ_THRESHOLD_EXPERIMENT_ID=<ID>
jupyter lab notebooks/08_similarity_reflection_threshold_plots.ipynb
```

## Regras para evitar mistura de resultados

- Não renomeie o diretório `<ID>` nem edite os JSONL manualmente.
- Não use `--fresh` após uma queda; ele existe somente para reiniciar
  deliberadamente o estágio correspondente.
- Não rode novamente `run_similarity_threshold.py` sobre o mesmo resultado
  depois do `finish`, pois ele recria a análise-base sem a condição externa.
  Se isso acontecer, basta executar `finish` novamente; os caches serão usados.
- Antes de `finish`, use `status` para conferir quantas respostas têm texto
  utilizável. Um receipt incompleto é aceito e auditado por padrão.
- Resultados finais e plots continuam locais no servidor de GPU; somente o
  pacote de intercâmbio é versionado.
