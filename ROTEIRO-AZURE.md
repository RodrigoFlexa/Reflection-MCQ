# Roteiro: professor GPT-5 via Azure OpenAI

Marque cada `- [ ]` conforme concluir. Cada passo traz o **comando completo** e o
**critério de aceite** — o que você deve ver na tela se deu certo.

**Legenda das máquinas:**

| | onde | o que faz |
|---|---|---|
| 🖥️ **LOCAL** | esta máquina (`/home/rodrigo.flexa/Reflection-MCQ`) | GPU, modelos pequenos, baseline, eval, análise |
| 🏢 **PTB** | máquina Petrobras | sem GPU, sem pesos; só Azure OpenAI e as reflexões |

**A ideia em uma frase:** o git carrega, numa branch dedicada, as *perguntas* e as
*respostas dos alunos* daqui para lá; lá o GPT-5 escreve as reflexões; o git
traz as reflexões de volta; e aqui os modelos pequenos respondem o teste com
essas reflexões no prompt.

A máquina Petrobras **só roda a etapa `reflect`**. Sem GPU, sem pesos, sem
baseline, sem eval — os três comandos que existem lá são `import-bundle`,
`reflect` e `export-bundle`.

```
🖥️ LOCAL                                         🏢 PTB
baseline (train) ──┐                              ┌── reflect (gpt5)
                   │  exchange/to-azure/          │
                   └────────► git ────────────────┘
                                                  │
       eval + analyze ◄────── git ◄───────────────┘
                             exchange/from-azure/
```

**Volume:** 4 alunos × 2 profundidades × 1.763 questões de treino =
**14.104 chamadas ao GPT-5**. O passo 6 é um piloto obrigatório justamente por isso.

> O `gpt4` continua registrado em `rmcq/config.py` e funciona do mesmo jeito —
> basta trocar `gpt5` por `gpt4` (ou passar os dois) quando você quiser. Esta
> rodada é só GPT-5.

---

## Duas armadilhas que valem a leitura antes de começar

> **1. `RMCQ_ACTIVE_MODELS` precisa listar os 6 modelos nas duas máquinas.**
> `--teachers gpt5` é validado contra a lista de professores ativos, que sai de
> `RMCQ_ACTIVE_MODELS`. Se `gpt5` não estiver lá, o comando morre com
> "professor desconhecido". Use sempre:
> `RMCQ_ACTIVE_MODELS=phi4-mini,llama3-8b,qwen3-8b,mistral-7b,gpt5,gpt4`

> **2. Na PTB, o `.env` precisa de `RMCQ_TEACHERS=gpt5` e `RMCQ_API_ONLY=1`.**
> Sem o primeiro, `reflect` sem `--teachers` viraria *todos* os professores
> ativos — incluindo os quatro modelos locais, que aquela máquina tentaria
> carregar sem GPU e sem pesos. O segundo é a rede de segurança: com ele, pedir
> um modelo local falha na hora, com uma mensagem que diz o porquê.

---

## Fase 0 — Pré-voo 🖥️ LOCAL

- [ ] **0.1** Conferir que os baselines de treino dos 4 alunos existem (é o insumo que vai viajar)

```bash
cd /home/rodrigo.flexa/Reflection-MCQ
for m in phi4-mini llama3-8b qwen3-8b mistral-7b; do
  echo -n "$m: "; ls results/baseline/$m/*_train.jsonl 2>/dev/null | wc -l
done
```

> ✅ **Aceite:** `5` para cada um dos quatro modelos (5 datasets).
> Se algum der `0`, gere antes: `python -m rmcq baseline --students <modelo> --splits train`

- [ ] **0.2** Conferir estado do git e que o `.env` não está rastreado

```bash
git status --short
git check-ignore -v .env
```

> ✅ **Aceite:** a segunda linha imprime `.gitignore:3:.env	.env`.
> Se o `.env` aparecer em `git status`, **pare** e resolva antes — ele vai
> conter a chave do Azure.

- [ ] **0.3** Fixar os modelos ativos no `.env` local

```bash
grep -n RMCQ_ACTIVE_MODELS .env
```

Edite a linha para:

```
RMCQ_ACTIVE_MODELS=phi4-mini,llama3-8b,qwen3-8b,mistral-7b,gpt5,gpt4
```

> ✅ **Aceite:** `python -m rmcq status` lista `gpt5` e `gpt4` entre os professores.
> Aqui eles não precisam de credencial nenhuma: existem só para que `eval` e
> `analyze` saibam ler as reflexões que voltarem.

---

## Fase 1 — Validar o código novo sem gastar API 🖥️ LOCAL

- [ ] **1.1** Backends disponíveis

```bash
python -c "from rmcq.backends import available_backends; print(available_backends())"
```

> ✅ **Aceite:** `azure` aparece como `indisponível: falta AZURE_OPENAI_ENDPOINT...`
> — nesta máquina isso está **correto**, é lá que a credencial mora.

- [ ] **1.2** Ensaio com o backend falso (não toca no Azure, não custa nada)

```bash
python -m rmcq reflect --students phi4-mini --teachers gpt5 \
  --depths simple --datasets arc --limit 3 --backend stub
```

> ✅ **Aceite:** cria `results/reflections/phi4-mini__gpt5__simple/arc.jsonl` com
> 3 linhas. Confira que o papel ficou certo:
> ```bash
> python -c "import json; r=json.loads(open('results/reflections/phi4-mini__gpt5__simple/arc.jsonl').readline()); print(r['condition'], r['reflection_perspective'], r['teacher_model'])"
> ```
> deve imprimir `external_reflection teacher gpt5`.

- [ ] **1.3** Apagar o ensaio, para não misturar dado falso com dado real

```bash
rm -rf results/reflections/phi4-mini__gpt5__simple
```

---

## Fase 2 — Empacotar e enviar 🖥️ LOCAL

- [ ] **2.1** Gerar o pacote de ida

```bash
python -m rmcq export-bundle --direction to-azure
```

> ✅ **Aceite:** `pacote to-azure: 25 arquivos, 8815 linhas` (5 splits de treino +
> 4 alunos × 5 datasets), ~16 MB em `exchange/to-azure/`.

- [ ] **2.2** Criar a branch de troca e enviar

```bash
git checkout -b azure-exchange
git add exchange/to-azure rmcq/ sanity_gpt5.py requirements-azure.txt .env.example ROTEIRO-AZURE.md
git status --short          # confira: NENHUM .env na lista
git commit -m "Adiciona professores Azure OpenAI e pacote de troca to-azure"
git push -u origin azure-exchange
```

> ✅ **Aceite:** o push conclui e `git status --short` sai vazio.
> ⚠️ Se `.env` aparecer no `git status`, **não commite** — rode `git reset .env`.

---

## Fase 3 — Preparar a máquina Petrobras 🏢 PTB

- [ ] **3.1** Clonar e entrar na branch de troca

```bash
git clone https://github.com/RodrigoFlexa/Reflection-MCQ.git
cd Reflection-MCQ
git checkout azure-exchange
```

> ✅ **Aceite:** `ls exchange/to-azure/` mostra `manifest.json`, `baseline/`, `splits/`.

- [ ] **3.2** Ambiente Python enxuto (sem torch, sem CUDA, sem vLLM)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-azure.txt
```

> ✅ **Aceite:** instala apenas `openai`, `tqdm`, `tiktoken` e dependências.
> Se a rede exigir espelho interno, acrescente `-i <url-do-espelho>`.
> Verificação de que nada pesado entra:
> ```bash
> python -c "import rmcq.stages.reflect, sys; print([m for m in ('torch','vllm','transformers') if m in sys.modules] or 'nenhum modulo pesado')"
> ```
> deve imprimir `nenhum modulo pesado`.

- [ ] **3.3** Criar o `.env` com as credenciais Azure

```bash
cp .env.example .env
```

Edite `.env` e preencha **só** estas linhas (o resto pode ficar como está):

```
AZURE_OPENAI_ENDPOINT=https://<seu-recurso>.openai.azure.com/
AZURE_OPENAI_API_KEY=<sua-chave>
AZURE_OPENAI_API_VERSION=2024-10-21

RMCQ_AZURE_DEPLOYMENT_GPT5=<nome-exato-do-deployment-gpt5>

RMCQ_ACTIVE_MODELS=phi4-mini,llama3-8b,qwen3-8b,mistral-7b,gpt5,gpt4
RMCQ_BACKEND=azure
RMCQ_AZURE_CONCURRENCY=4

# Esta máquina só gera reflexões, e só com o GPT-5:
RMCQ_TEACHERS=gpt5
RMCQ_API_ONLY=1
```

> As duas últimas linhas são o que torna esta máquina segura de operar:
> `RMCQ_TEACHERS=gpt5` faz `reflect` sem argumento já significar "gpt5", e
> `RMCQ_API_ONLY=1` recusa qualquer modelo local com uma mensagem explicando o
> motivo, em vez de tentar carregar pesos que não existem aqui.

> ⚠️ **O nome do deployment não é o nome do modelo.** Deployments corporativos
> costumam ter prefixo próprio (`fgl-gpt-5-prod`). Pegue o nome exato no portal
> do Azure OpenAI, em *Deployments*.
> ✅ **Aceite:** `git status --short` **não** lista o `.env` (o `.gitignore` já o cobre).

- [ ] **3.4** Confirmar que a credencial funciona

```bash
python -c "from rmcq.backends import available_backends; print(available_backends()['azure'])"
```

> ✅ **Aceite:** `ok (openai <versão>)`.
> ❌ Se disser `falta AZURE_OPENAI_...`, o `.env` não foi lido ou tem a variável vazia.

- [ ] **3.5** Confirmar que a máquina está no modo "só reflexões"

```bash
python -c "
from rmcq.config import TEACHERS, STUDENTS, API_ONLY
print('professores:', TEACHERS)
print('alunos     :', STUDENTS)
print('só API     :', API_ONLY)"
```

> ✅ **Aceite:** `professores: ('gpt5',)`, os 4 alunos listados, `só API: True`.
> Os alunos aparecem porque as respostas deles vêm prontas no pacote — nenhum
> peso é carregado.

---

## Fase 4 — Materializar os dados 🏢 PTB

- [ ] **4.1** Conferir e importar o pacote

```bash
python -m rmcq import-bundle --direction to-azure
```

> ✅ **Aceite:** `25 arquivos importados: 8815 linhas novas`.
> O comando confere sha256 e contagem de linhas de cada arquivo **antes** de
> materializar. Se abortar citando sha256, o transporte corrompeu algo — refaça
> o `export-bundle` na máquina local.
> Isso não é zelo: uid divergente faria a reflexão casar com a pergunta errada,
> e nada no pipeline acusaria depois.

- [ ] **4.2** Conferir que os arquivos caíram nos caminhos do pipeline

```bash
ls data/splits/*/train.jsonl
ls results/baseline/*/
```

> ✅ **Aceite:** 5 arquivos `train.jsonl` e 4 diretórios de aluno com 5 arquivos cada.

---

## Fase 5 — Teste de fumaça do Azure 🏢 PTB

- [ ] **5.1** Uma pergunta, um modelo, para validar a ligação

```bash
python -m rmcq smoke --models gpt5
```

> ✅ **Aceite:** responde `B` (Júpiter).
> ❌ **Resposta vazia com `finish_reason='length'`** → o GPT-5 gastou o orçamento
> pensando. Suba no `.env`: `RMCQ_AZURE_REASONING_MIN_TOKENS=8000` e repita.
> ❌ **`DeploymentNotFound`** → `RMCQ_AZURE_DEPLOYMENT_*` está com nome errado.

---

## Fase 6 — Piloto pago: o teste de sanidade 🏢 PTB

**Não pule.** São 14 mil chamadas na grade cheia; esta é a hora barata de
descobrir que o GPT-5 está escrevendo a coisa errada.

- [ ] **6.1** Rodar o script de sanidade sobre uma resposta que o aluno ERROU

```bash
python sanity_gpt5.py -n 3 --only-wrong
```

O script pega questões reais de treino, a resposta que o `phi4-mini` de fato
deu no baseline, monta **o mesmo prompt** que a etapa `reflect` montaria, chama
o GPT-5 e imprime tudo lado a lado. Ele **não grava nada** em `results/` — é
diagnóstico, não dado experimental.

> ✅ **Aceite:** para cada questão você vê pergunta, opções, gabarito, resposta
> do aluno, e a reflexão do GPT-5, com contagem de palavras/frases/tokens ao fim.
> O script termina com código 1 se alguma reflexão vier vazia.

- [ ] **6.2** Ler as três reflexões e bater o checklist que o script imprime

O próprio script lista o que conferir. Em particular:

| conferir | por quê |
|---|---|
| fala do raciocínio **deste** aluno | reflexão genérica não ensina nada; é o ponto do experimento |
| **não** revela a alternativa correta | o prompt proíbe, mas modelo de reasoning às vezes desobedece — e se revelar, o `eval` vira vazamento de gabarito |
| 3 a 6 frases no modo `simple` | se vier muito curta, `RMCQ_AZURE_MAX_TOKENS` está apertado |
| nenhuma vazia | vazia = erro de configuração, nunca abstenção |

> ⛔ **Se a reflexão entregar a resposta certa, pare aqui.** Isso contamina o
> experimento inteiro: no `eval`, o aluno leria o gabarito em vez de raciocinar.
> Ajuste o prompt em `rmcq/common.py` antes de gastar os 14 mil.

- [ ] **6.3** Conferir também o caso de acerto e a profundidade `complex`

```bash
python sanity_gpt5.py -n 2 --only-right
python sanity_gpt5.py -n 2 --depth complex --only-wrong
```

> ✅ **Aceite:** no caso de acerto a reflexão explica *por que a abordagem
> funcionou* (não inventa um erro); em `complex` o texto é visivelmente mais
> longo e analítico que em `simple`.

> 💡 Outras opções úteis: `--student llama3-8b`, `--dataset gsm8k`,
> `--show-prompt` (imprime o prompt exato enviado), `--full-answer`,
> `--backend stub` (não chama a API, só mostra o formato).

---

## Fase 7 — Grade completa 🏢 PTB

- [ ] **7.1** Estimar antes de rodar

```bash
python -m rmcq reflect --dry-run
```

Sem `--teachers` porque `RMCQ_TEACHERS=gpt5` no `.env` já resolve isso.

> ✅ **Aceite:** o cabeçalho mostra exatamente:
> ```
> configurações                  40
> itens totais               14,104
> pendentes                  14,104
> tokens de entrada       7,334,080  (estimado)
> tokens de saída         4,019,640  (estimado)
> ```
> Se aparecer 80 configurações, `RMCQ_TEACHERS` não pegou e o `gpt4` entrou junto.
>
> **Compare com o orçamento que você tem no Azure antes de seguir.** Multiplique
> pelo preço por 1M de tokens do seu contrato — e lembre que num modelo de
> reasoning os tokens de raciocínio contam como saída, então 4M é piso, não teto.
>
> ⚠️ As duas linhas de "tempo em vLLM / transformers" **não valem aqui** — são
> estimativas de GPU local. Pelo Azure, com `RMCQ_AZURE_CONCURRENCY=4` e ~4 s por
> chamada, a conta é `14.104 ÷ 4 × 4 s ≈ 4 h`.
>
> 💡 Para cortar o custo pela metade, rode só `--depths simple`: são 7.052
> chamadas, e já dá para comparar professor grande contra professor pequeno.

- [ ] **7.2** Rodar, numa sessão que sobrevive à desconexão

```bash
tmux new -s reflect          # ou: nohup ... &

python -m rmcq reflect --log-file
```

> ✅ **Aceite:** a barra de progresso avança e o log vai para `results/logs/`.
> Solte com `Ctrl-B D`; volte com `tmux attach -t reflect`.
>
> 💡 **Se cair no meio:** rode o **mesmo comando** de novo. Duas proteções agem
> juntas — o JSONL é retomável por uid (não regera o que já está gravado) e o
> cache em disco (`results/cache/azure/`) devolve de graça as chamadas já pagas
> do lote interrompido.
>
> 💡 **Se aparecerem muitos 429:** baixe `RMCQ_AZURE_CONCURRENCY` para `2` no `.env`.
>
> 💡 **Para fatiar o gasto:** rode por dataset, ex.
> `python -m rmcq reflect --datasets arc openbookqa`.

- [ ] **7.3** Conferir a colheita

```bash
for d in results/reflections/*gpt5*/; do echo -n "$d "; cat $d/*.jsonl | wc -l; done
```

> ✅ **Aceite:** 8 diretórios (4 alunos × 2 profundidades), 1.763 linhas cada.

---

## Fase 8 — Devolver as reflexões 🏢 PTB

- [ ] **8.1** Empacotar

```bash
python -m rmcq export-bundle --direction from-azure
```

> ✅ **Aceite:** `pacote from-azure: 40 arquivos, 14104 linhas` — 8 diretórios
> (4 alunos × 2 profundidades) × 5 datasets. Sem `--teachers` porque
> `RMCQ_TEACHERS=gpt5` já resolve. Se o log avisar de combinações faltando,
> volte à Fase 7.

- [ ] **8.2** Commitar e enviar

```bash
git add exchange/from-azure
git status --short          # confira: NENHUM .env na lista
git commit -m "Reflexões geradas por GPT-5"
git push origin azure-exchange
```

> ✅ **Aceite:** push concluído.
> ⚠️ Se o `.env` aparecer, `git reset .env` antes de commitar. **A chave do Azure
> não pode ir para o GitHub.**

---

## Fase 9 — Consumir as reflexões 🖥️ LOCAL

- [ ] **9.1** Trazer e materializar

```bash
cd /home/rodrigo.flexa/Reflection-MCQ
git checkout azure-exchange
git pull origin azure-exchange
python -m rmcq import-bundle --direction from-azure
```

> ✅ **Aceite:** `40 arquivos importados: 14104 linhas novas`.
> O sha256 de cada arquivo é conferido antes de materializar.

- [ ] **9.2** Conferir que chegaram onde o `eval` procura

```bash
ls results/reflections/ | grep gpt5
```

> ✅ **Aceite:** 8 diretórios `{aluno}__gpt5__{simple|complex}`.

- [ ] **9.3** Índice de similaridade (pule se já existe)

```bash
python -m rmcq index
```

> ✅ **Aceite:** "reaproveitados" lista os 5 datasets, se o índice já estava pronto.
> O índice não depende do professor — é o mesmo dos experimentos anteriores.

- [ ] **9.4** Ensaio do `eval` antes da rodada longa

```bash
python -m rmcq eval --students phi4-mini --teachers gpt5 --depths simple \
  -k 3 --datasets arc --limit 10 --backend stub
```

> ✅ **Aceite:** `gpt5/simple/k3/arc: 10 respostas, acerto N%` sem erro de
> reflexão ausente. Depois: `rm -rf results/eval/phi4-mini__gpt5__simple__k3`.

- [ ] **9.5** Avaliação de verdade (usa a GPU local)

```bash
python -m rmcq eval --teachers gpt5 --log-file
```

> ✅ **Aceite:** um diretório por configuração em `results/eval/`.
> ⏱️ É a etapa mais longa do lado local — use `tmux`.

- [ ] **9.6** Análise

```bash
python -m rmcq analyze --teachers gpt5
cat results/analysis/summary.md
```

> ✅ **Aceite:** `results/analysis/accuracy.csv` e `utility.csv` ganham linhas com
> `teacher_model` = `gpt5`, comparáveis lado a lado com os professores pequenos.
> **É esta comparação que responde a pergunta do experimento:** um professor
> grande produz reflexões mais úteis para um aluno pequeno do que outro modelo pequeno?

---

## Fase 10 — Fechar 🖥️ LOCAL

- [ ] **10.1** Levar tudo para `dev`

```bash
git checkout dev
git merge azure-exchange
git push origin dev
```

> ✅ **Aceite:** merge sem conflito — os dois lados escrevem em arquivos
> disjuntos (`exchange/to-azure/` aqui, `exchange/from-azure/` lá).

- [ ] **10.2** (Opcional) Aliviar o repositório

O pacote de troca já cumpriu a função; os dados vivem em `results/` e
`data/splits/`. Se quiser tirar os ~48 MB do rastreamento:

```bash
git rm -r --cached exchange/
echo "exchange/" >> .gitignore
git commit -m "Retira o pacote de troca do rastreamento"
```

> ⚠️ Só faça isso **depois** que ambos os lados importaram. E note que o histórico
> continua guardando os arquivos — para valer, seria preciso reescrever o histórico.

---

## Referência rápida de problemas

| Sintoma | Causa provável | Correção |
|---|---|---|
| `professor desconhecido: ['gpt5']` | `RMCQ_ACTIVE_MODELS` sem `gpt5` | acrescente `gpt5,gpt4` no `.env` |
| `credencial do Azure ausente` | `.env` não lido, ou variável vazia | rode a partir da raiz do repo; confira `AZURE_OPENAI_ENDPOINT` |
| `resposta vazia ... finish_reason='length'` | reasoning gastou o orçamento pensando | `RMCQ_AZURE_REASONING_MIN_TOKENS=8000` |
| `DeploymentNotFound` | nome do deployment errado | pegue o nome exato no portal do Azure |
| muitos 429 / execução lenta | concorrência alta demais | `RMCQ_AZURE_CONCURRENCY=2` |
| `pacote não confere (sha256)` | transporte corrompeu o arquivo | refaça o `export-bundle` na origem |
| `gateway rejeitou 'seed'` (aviso) | deployment não aceita o parâmetro | nenhuma — o código remove e segue |
| tentaria carregar Llama-3 na PTB | `RMCQ_TEACHERS` não definido | ponha `RMCQ_TEACHERS=gpt5` e `RMCQ_API_ONLY=1` no `.env` da PTB |
| `RMCQ_API_ONLY=1: esta máquina só roda modelos de API` | pediu modelo local na PTB | é a proteção agindo — use `--teachers gpt5` |
| a reflexão entrega o gabarito | GPT-5 desobedeceu o prompt | **pare**; ajuste `REFLECTION_PROMPTS` em `rmcq/common.py` |

---

## O que cada peça faz

| Arquivo | Papel |
|---|---|
| [rmcq/backends/azure.py](rmcq/backends/azure.py) | cliente Azure OpenAI: reasoning vs chat, retry, cache, vazio = falha |
| [rmcq/stages/exchange.py](rmcq/stages/exchange.py) | empacota/verifica/materializa o que atravessa por git |
| [rmcq/config.py](rmcq/config.py) | registro de `gpt5`/`gpt4` e as variáveis `RMCQ_AZURE_*` |
| [rmcq/backends/__init__.py](rmcq/backends/__init__.py) | roteia modelo de API para o Azure, ignorando `--backend` |
| [sanity_gpt5.py](sanity_gpt5.py) | teste de sanidade: imprime reflexões reais do GPT-5 |
| [requirements-azure.txt](requirements-azure.txt) | dependências da máquina PTB (sem torch) |
| [guia-azure-openai-fgl.md](guia-azure-openai-fgl.md) | o guia que o backend implementa |
