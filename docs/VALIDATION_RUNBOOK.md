# Validação: GPU primeiro; professor opcional depois

Execute na pasta `Reflection-MCQ`. Branch de compartilhamento: `lean-backends`.
A GPU padrão dos comandos é 3; altere `--gpu` se necessário. Os estudantes rodam
sequencialmente dentro de cada processo, com batching do vLLM. Nunca carregamos
dois modelos ao mesmo tempo na mesma GPU.

Há dois modos: **uma GPU** (seção 1) roda os nove estudantes em sequência;
**duas GPUs** (seção 1b) divide a grade em duas partições que rodam ao mesmo
tempo, em GPUs diferentes, e depois são unidas por `merge`. As duas produzem a
mesma run, com o mesmo ID e os mesmos resultados.

## Grade congelada

| Nome pedido | Chave nos resultados | Checkpoint HF usado pelo vLLM |
|---|---|---|
| phi2 | phi2 | microsoft/phi-2 |
| deepseek-r1:8b | deepseek-r1-8b | deepseek-ai/DeepSeek-R1-Distill-Llama-8B |
| deepseek-r1:1.5b | deepseek-r1-1.5b | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| llama3.1:8b | llama3.1-8b | meta-llama/Llama-3.1-8B-Instruct |
| llama3.2:3b | llama3.2-3b | meta-llama/Llama-3.2-3B-Instruct |
| qwen2.5:3b | qwen2.5-3b | Qwen/Qwen2.5-3B-Instruct |
| qwen2.5:7b | qwen2.5-7b | Qwen/Qwen2.5-7B-Instruct |
| ministral-3:3b | ministral-3-3b | mistralai/Ministral-3-3B-Instruct-2512-BF16 |
| ministral-3:8b | ministral-3-8b | mistralai/Ministral-3-8B-Instruct-2512-BF16 |

Os nomes Ollama identificam a família solicitada; não usamos o servidor Ollama
nem seus arquivos GGUF.

**O DeepSeek-R1 é padronizado na destilação sobre Llama.** Onde há escolha, é
essa a base, e os nomes canônicos (`deepseek-r1-8b`) escondem a destilação
porque ela é constante.

Duas consequências que valem estar escritas:

- **O 1.5B é exceção forçada.** A família Distill-Llama oficial tem 8B e 70B, e
  nada entre os dois: `DeepSeek-R1-Distill-Llama-1.5B` **não existe**. As
  destilações são Qwen 1.5B/7B/14B/32B e Llama 8B/70B (índice do Hugging Face,
  13/09/2026). Então `deepseek-r1-1.5b` fica sobre Qwen por falta de
  alternativa, não por escolha.
- **`DeepSeek-R1-0528-Qwen3-8B` saiu da grade.** É outra base. O run de
  validação v5 rodou esse checkpoint no lugar do DeepSeek de 8B, então essas
  22.461 linhas **não contam** e o modelo precisa ser gerado do zero. A chave
  continua registrada em `rmcq.config.MODELS` só para que aquele run continue
  legível — é dele que saem os outros oito alunos. Pesos HF ausentes no cache são baixados no primeiro uso.
Os Ministral usam os checkpoints oficiais BF16 para seguir a precisão dos demais
estudantes, com tokenizer/config/load no formato Mistral e entradas somente textuais.
Isso exige vLLM >= 0.12.0 e mistral-common >= 1.8.6.

Fontes: [DeepSeek atual](https://ollama.com/library/deepseek-r1:8b),
[Ministral 3B BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16),
[Ministral 8B BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16),
[instruções vLLM da Mistral](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512).

### Professores

| Professor | Ensina | Backend |
|---|---|---|
| gpt-5-4-petrobras | os 9 alunos | Azure, servidor Petrobras |
| deepseek-r1-8b | os 5 de até 3B | vLLM, GPU local |
| llama3.1-8b | os 5 de até 3B | vLLM, GPU local |
| qwen2.5-7b | os 5 de até 3B | vLLM, GPU local |
| ministral-3-8b | os 5 de até 3B | vLLM, GPU local |

Alunos de até 3B: phi2, deepseek-r1-1.5b, llama3.2-3b, qwen2.5-3b,
ministral-3-3b. Os professores abertos não ensinam modelos do próprio porte —
um 8B instruindo outro 8B mede transferência entre pares, que é outra pergunta.

São 29 pares professor-aluno. A lista canônica está em
[`rmcq/grid.py`](../rmcq/grid.py); mudar a grade é mudar esse arquivo, e o
pipeline, a auditoria e os notebooks acompanham sozinhos.

Diferente do desenho anterior, **quatro dos cinco professores rodam na GPU
local**, junto dos alunos. Só o GPT-5.4 depende do servidor Petrobras, e a
etapa 3 continua sendo o caminho dele.

Datasets: **aqua, arc, logiqa2, openbookqa, race**, sempre treino → validação.
Recuperação top-1 por similaridade, permitindo repetir uma fonte de treino entre
questões de validação. Cada fonte única é respondida/refletida uma vez por modelo.
Não aplicamos threshold durante a geração; ele será estudado depois, sem regenerar.

Mantemos prompts e perfil `final` do último teste. Limites resposta/reflexão:
Phi-2 512/768; ambos DeepSeek 4096/4096; demais estudantes 1024/1024.
Contexto operacional: Phi-2 2048; demais 8192. Não há recorte do prompt de
transferência nem retry para esconder falhas. O orçamento da reflexão do Phi-2
pode ser reduzido ao espaço restante, com marcação. Thinking é retirado da saída
utilizável, preservando texto bruto e auditoria; os R1 não têm um desligamento
de reasoning garantido. O avaliador auxiliar continua Llama-3.1-8B via vLLM.

## 0. O caminho curto: um comando por split

```bash
python run_local.py --split validation --gpu 0
python run_local.py --split test       --gpu 1
```

Percorre prepare → self-eval → teacher (os quatro abertos) → finish →
consolidação, pulando o que já existe. O GPT-5.4 fica de fora por padrão e
entra depois com `--with-external-teacher`, ou pela etapa 3b. As seções abaixo
descrevem os mesmos estágios um a um, para quando for preciso intervir no meio.

## 1. GPU: instalar o ambiente e iniciar tudo localmente

Um ambiente separado evita modificar a pilha usada nas runs anteriores. Use um
Python suportado pela versão de vLLM instalada. Os arquivos canônicos de AQuA,
ARC, LogiQA2 e OpenBookQA devem ser os já usados neste servidor; o comando abaixo
instala/verifica o RACE. Mantenha `.env` e as permissões HF dos Llama nesse servidor.

```bash
git fetch origin
git switch lean-backends
git pull --ff-only origin lean-backends
python3 -m venv .venv-validation
source .venv-validation/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/validation.txt
python tools/prepare_datasets.py --datasets race
python validation_ops.py start local --gpu 3
```

`start local` roda em segundo plano e continua após desconectar do SSH:

1. verifica os dez arquivos treino/validação e a GPU;
2. carrega cada um dos nove modelos em um processo separado e faz uma geração
   curta de verificação de vLLM/tokenizer/VRAM (essas saídas não entram na análise);
3. prepara os pares, respostas de treino e autorreflexões;
4. avalia baseline, self_simple e self_complex na validação;
5. gera tabelas e figuras da validação, inicialmente com fallback **False**.

Nenhuma chamada GPT/Azure é feita. Se algum checkpoint não puder carregar,
o processo para com o erro; não troca para HF/Ollama nem pula o modelo.

```bash
python validation_ops.py status
tail -n 50 .run_state/validation-local.log
```

O ID aparece no log e fica em `.run_state/validation_experiment_id` quando o
prepare termina. Enquanto prepara, também está em `.run_state/experiment_id`.
Para verificar a fila antes de o ID existir, consulte o log ou
`python experiment_ops.py status`.

## 1b. Duas GPUs em paralelo

A divisão é por modelo, porque o vLLM já carrega um motor por modelo: nada passa
a ser carregado duas vezes e cada artefato de geração já é gravado por modelo.
A grade congelada continua sendo a mesma nove; só o trabalho é dividido.

| Partição | Estudantes | Peso |
|---|---|---|
| `p1` | deepseek-r1-8b, phi2, llama3.2-3b, qwen2.5-3b, ministral-3-3b | 1 modelo 8B com thinking + 4 pequenos |
| `p2` | deepseek-r1-1.5b, llama3.1-8b, qwen2.5-7b, ministral-3-8b | 3 modelos 8B + 1 pequeno com thinking |

Para mudar quem fica em cada lado, edite `VALIDATION_PARTITIONS` em
`run_experiment.py`; um teste falha se a divisão deixar de cobrir a grade
exatamente uma vez. O juiz fixo (llama3.1-8b) é carregado nas duas GPUs, porque
cada processo resolve as próprias respostas não parseadas.

Abra dois terminais no **mesmo checkout**, cada um com o ambiente ativado:

```bash
# terminal 1
python validation_ops.py start local --p1 --gpu 3

# terminal 2
python validation_ops.py start local --p2 --gpu 7
```

O ID da run é derivado da configuração congelada dos nove modelos, então as duas
partições calculam o mesmo ID e escrevem na mesma run. Quem chegar primeiro
calcula a recuperação top-1; a outra espera nesse ponto e reaproveita os mesmos
pares, sem embutir o corpus duas vezes. Cada partição tem o próprio lock, o
próprio `job.<partição>.json` e o próprio log:

```bash
python validation_ops.py status
tail -n 80 .run_state/validation-local.p1.log
tail -n 80 .run_state/validation-local.p2.log
```

Nenhuma das duas escreve `prepare_receipt.json` nem `self_eval_receipt.json`:
cada uma escreve o recibo da própria partição. Isso é proposital, para que uma
run pela metade nunca pareça inteira e para que nenhuma GPU espere a outra na
fronteira entre prepare e self-eval. Quando as duas terminarem:

```bash
python validation_ops.py merge
```

`merge` não gera nada: confere que os nove modelos estão presentes, concatena
`models/<modelo>/validation.jsonl` em `all_outcomes.jsonl`, recalcula
`accuracy.csv`, escreve os recibos da run inteira, **promove o resultado para
`results_definitivos/validation/`** e roda a análise. Se faltar
alguma partição, ele diz quais modelos está esperando e não escreve recibo. O
restante do roteiro (seções 2 a 4) não muda; `share`, `teacher` e `finish`
continuam agindo sobre a run inteira e não aceitam `--part`.

Retomada: se uma partição falhar, repita apenas o `start local` dela, com a
mesma GPU. As chamadas já concluídas vêm do cache e a outra partição não é
afetada.

## 2. Analisar e compartilhar a etapa parcial

Os resultados brutos desta etapa ficam em
`data/results/reflection_top1/<id>/self_eval/`. O recibo é `self_eval_receipt.json`;
isso **não** é um experimento com professor concluído.

O número que vale não sai daí, e sim do arquivo canônico:

```bash
python consolidate_results.py build --split validation
python consolidate_results.py status
```

Isso escreve `results_definitivos/validation/` com o `all_outcomes.jsonl`
fundido, a proveniência de cada linha em `manifest.json`, a cobertura por
aluno × condição × professor × dataset em `coverage.csv` e, em `gaps.json`, o
que a grade ainda pede. **`gaps.json` é a resposta para "o que falta rodar?"**,
separando célula vazia (trabalho inteiro) de célula parcial (retomada).

Abra `notebooks/validation_accuracy_by_similarity.ipynb`: ele lê o arquivo
canônico direto, sem id de run. O booleano de fallback, os filtros, a grade de
thresholds, o mínimo de amostras e a cobertura ficam na primeira célula.

Para repetir os plots sem abrir o notebook:

```bash
python validation_ops.py analyze
```

Compartilhe as duas partes ao concluir o trabalho local:

```bash
python validation_ops.py share prepare
python validation_ops.py share self-eval
```

Cada comando verifica o recibo, cria pacotes com hashes e partes de até 20 MiB,
faz commit somente desses pacotes e push para a branch atual. Não inclui `.env`,
pesos ou caches de geração. Os pares contêm os itens de treino/validação necessários;
o pacote prepare também contém os três splits canônicos do RACE.
`current_validation.json` mantém a seleção separada da última run de teste.

Em outra máquina, para abrir apenas a análise parcial:

```bash
git pull --ff-only origin lean-backends
python validation_ops.py restore self-eval
python -m pip install -r requirements/analysis.txt
python validation_ops.py analyze
```

Para exportar com fallback True, usando o ID exibido por `status`:

```bash
python analyze_validation.py --fallback
```

## 3. Professores

São cinco, e eles não rodam no mesmo lugar: o GPT-5.4 precisa da credencial
Azure, os quatro abertos precisam de GPU. `--only-teachers` diz quais gerar
nesta máquina; o recibo só fica completo quando todos tiverem passado, e cada
lado soma ao que o outro já deixou no disco.

### 3a. GPU: os quatro professores abertos

```bash
python validation_ops.py start teacher --gpu 3 \
  --only-teachers deepseek-r1-8b,llama3.1-8b,qwen2.5-7b,ministral-3-8b
python validation_ops.py status
```

Cada professor aberto reflete sobre os cinco alunos de até 3B. Um professor cujas
reflexões já estão completas no disco não carrega engine nenhum, então repetir o
comando depois de uma interrupção retoma por par professor-aluno.

### 3b. Petrobras: o GPT-5.4

No ambiente Python do Petrobras, com as credenciais Azure já configuradas:

```bash
git fetch origin
git switch lean-backends
git pull --ff-only origin lean-backends
python validation_ops.py restore prepare
python -m pip install -r requirements/azure.txt
python validation_ops.py start teacher --only-teachers gpt-5-4-petrobras
```

GPT-5.4 ensina os nove alunos. Recebe somente questões/tentativas de **treino** dos estudantes e o
feedback já calculado. Ele gera reflexões simples e complexas. Não responde
treino por conta própria, não faz autorreflexão própria, não avalia estudantes
como juiz e **não responde nenhuma questão de validação**.

```bash
python validation_ops.py status
tail -n 50 .run_state/teacher.log
```

Quando terminar:

```bash
python validation_ops.py share teacher
```

## 4. GPU: acrescentar as duas condições externas

Volte ao mesmo checkout GPU usado na etapa local; assim o snapshot parcial já
estará presente. Em outro checkout GPU, restaure também prepare e self-eval.

```bash
git pull --ff-only origin lean-backends
source .venv-validation/bin/activate
python validation_ops.py start finish --gpu 3
```

O comando importa os professores, gera somente teacher_simple e teacher_complex
— uma vez por professor de cada aluno — e reaproveita exatamente as linhas de
baseline/self já finalizadas. Exige o recibo de professor completo: sem todas as
reflexões, uma célula viraria linha `not_generated`, que depois se confunde com
falha real. Ao terminar, promove o resultado para `results_definitivos/` e
recria a análise. O snapshot `self_eval/` é preservado.
Se estiver em outro checkout, a sequência antes de `start finish` é:

```bash
python validation_ops.py restore prepare
python validation_ops.py restore self-eval
```

```bash
python validation_ops.py status
tail -n 50 .run_state/finish.log
```

Depois de concluído:

```bash
python validation_ops.py share finish
```

Na máquina de análise: `git pull --ff-only origin lean-backends` e
`python validation_ops.py restore finish`. O notebook passa a detectar as cinco
condições; continua sem linhas GPT estudante.

## Retomada e identidade

### Checkpoints com autorização

Dos nove estudantes, só os dois `meta-llama` são gated. Os demais são abertos:
Qwen 2.5 e Ministral 3 sob Apache 2.0, DeepSeek e Phi-2 sob MIT.

| Checkpoint | Precisa de autorização |
|---|---|
| meta-llama/Llama-3.1-8B-Instruct | sim, licença Llama 3.1 |
| meta-llama/Llama-3.2-3B-Instruct | sim, licença Llama 3.2, **pedido separado** |
| os outros sete | não |

A autorização é **por repositório**. Ter acesso ao Llama 3.1 não dá acesso ao
Llama 3.2. Peça em cada página, logado na mesma conta Hugging Face que emitiu o
`HF_TOKEN` do `.env` deste servidor. Para saber qual conta é essa:

```bash
huggingface-cli whoami
```

O preflight consulta o Hub antes de carregar qualquer peso e lista de uma vez
todos os repositórios que o token não consegue ler, com a URL de cada um. Sem
rede, ele apenas registra `NOTE: could not confirm Hub access` e segue, para não
travar um servidor que já tem tudo em cache.

#### Começar sem esperar a autorização

`--skip-gated` deixa de fora os checkpoints que o token ainda não lê, em vez de
parar a run:

```bash
python validation_ops.py start local --p1 --gpu 4 --skip-gated
```

O modelo pulado aparece como `SKIPPING <modelo>` no log e em `skipped_models`
no recibo da partição. Ele simplesmente não tem artefatos ainda.

Quando a autorização sair, **rode exatamente o mesmo comando**. O preflight
consulta o Hub de novo, encontra a lista de bloqueados vazia, e as etapas geram
só o que falta: um modelo cujo `students/<modelo>/train.jsonl` já cobre todas as
fontes é considerado pronto e nem tem o motor carregado de novo. Com a lacuna
preenchida, o `merge` fecha a run normalmente.

Enquanto faltar qualquer um dos nove, o `merge` recusa fechar a run e diz quais
modelos está esperando, e o `share` continua bloqueado. A flag é explícita de
propósito: deixar um modelo de fora de uma run científica não deve acontecer em
silêncio. Sem ela, a falta de autorização volta a parar a execução.

### Onde o preflight guarda o erro

Cada modelo é verificado em um processo próprio e a saída inteira é preservada.
Se algum falhar, o traceback completo vai para
`.run_state/validation_preflight_error.json` (ou `..._error.<partição>.json`),
junto com o comando de correção quando a falha é conhecida (FlashInfer,
falta de VRAM, checkpoint restrito). O log também imprime `DIAGNOSIS:` nessas
linhas, então `tail` curto já mostra o que fazer.

### FlashInfer: `array.array[int]` no Python 3.10/3.11

A causa é o próprio FlashInfer: `flashinfer/comm/fd_exchange.py` anota
`_fd_ancillary` com `tuple[tuple[int, int, array.array[int]]]`, e `array.array`
só aceita subscrito no Python 3.12+. Sem PEP 563, a anotação é avaliada ao
importar o módulo, então o vLLM nem chega a carregar pesos. A mensagem exata
`TypeError: 'type' object is not subscriptable` é a forma do Python 3.10
(no 3.11 ela nomeia `array.array`). O upstream corrigiu importando anotações
adiadas; o script abaixo aplica exatamente essa linha no ambiente ativo.

Execute no mesmo ambiente virtual GPU (neste exemplo, `venv`, GPU 7):

```bash
git pull --ff-only origin lean-backends
source venv/bin/activate
python tools/repair_flashinfer_annotations.py
python tools/repair_flashinfer_annotations.py --check
python validation_ops.py start local --gpu 7
python validation_ops.py status
tail -n 80 .run_state/validation-local.log
```

O script acrescenta `from __future__ import annotations`, como no
[código oficial do FlashInfer](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/fd_exchange.py).
Só modifica a assinatura conhecida dentro do ambiente ativo; preserva o arquivo
original em um backup adjacente e registra versão, caminho e hashes em
`.run_state/flashinfer_annotation_repair.json`. É idempotente, não reinstala
bibliotecas e não altera prompts, limites, judge ou kernels. A verificação local
dos nove modelos continua obrigatória. Se usar outro ambiente GPU depois, confira
essa compatibilidade nele também. O preflight agora detecta esse defeito antes
de carregar pesos e registra o hash do arquivo FlashInfer no relatório.

### Etapas interrompidas

- Falha durante a preparação: corrija o problema indicado e repita `start local`
  com a mesma configuração. As chamadas concluídas são recuperadas do cache.
- Prepare concluído, avaliação local interrompida: `python validation_ops.py start self-eval --gpu 3`.
- Professor interrompido: repita `python validation_ops.py start teacher`.
- Conclusão externa interrompida: repita `python validation_ops.py start finish --gpu 3`.
- Use `--experiment-id <ID>` com status/share/restore/analyze ou com start
  self-eval/teacher/finish para selecionar explicitamente uma run e evitar depender
  do ponteiro mais recente. Não use `--fresh` para a extensão opcional.
- Um lock local impede dois jobs iguais simultâneos no mesmo checkout. O lock é
  por partição: `p1` e `p2` rodam lado a lado de propósito, mas um segundo `p1`
  no mesmo checkout continua sendo recusado. Ele não coordena checkouts
  diferentes. Uma execução interrompida pode exigir esperar o worker encerrar
  antes de retomá-la.
- Os parâmetros científicos e fingerprints dos dados são congelados no manifesto.
  As versões de vLLM, torch, Transformers e mistral-common também entram na
  identidade da nova run e são conferidas nas etapas GPU seguintes.
  As etapas seguintes restauram os limites originais. Não misture ambientes ou
  revisões de código entre fases sem verificar compatibilidade.

Esta implementação foi validada localmente com backends simulados; a verificação
real de carga e geração dos nove modelos ocorre no preflight do servidor GPU.
