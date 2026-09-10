# Validação: GPU primeiro; professor opcional depois

Execute na pasta `Reflection-MCQ`. Branch de compartilhamento: `lean-backends`.
A GPU padrão dos comandos é 3; altere `--gpu` se necessário. Os nove estudantes
rodam sequencialmente, com batching do vLLM. Não carregamos os nove ao mesmo tempo.

## Grade congelada

| Nome pedido | Chave nos resultados | Checkpoint HF usado pelo vLLM |
|---|---|---|
| phi2 | phi2 | microsoft/phi-2 |
| deepseek-r1:8b atual | deepseek-r1-0528-qwen3-8b | deepseek-ai/DeepSeek-R1-0528-Qwen3-8B |
| deepseek-r1:1.5b | deepseek-r1-distill-qwen-1.5b | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| llama3.1:8b | llama3.1-8b | meta-llama/Llama-3.1-8B-Instruct |
| llama3.2:3b | llama3.2-3b | meta-llama/Llama-3.2-3B-Instruct |
| qwen2.5:3b | qwen2.5-3b | Qwen/Qwen2.5-3B-Instruct |
| qwen2.5:7b | qwen2.5-7b | Qwen/Qwen2.5-7B-Instruct |
| ministral-3:3b | ministral-3-3b | mistralai/Ministral-3-3B-Instruct-2512-BF16 |
| ministral-3:8b | ministral-3-8b | mistralai/Ministral-3-8B-Instruct-2512-BF16 |

Os nomes Ollama identificam a família solicitada; não usamos o servidor Ollama
nem seus arquivos GGUF. Pesos HF ausentes no cache são baixados no primeiro uso.
Os Ministral usam os checkpoints oficiais BF16 para seguir a precisão dos demais
estudantes, com tokenizer/config/load no formato Mistral e entradas somente textuais.
Isso exige vLLM >= 0.12.0 e mistral-common >= 1.8.6.

Fontes: [DeepSeek atual](https://ollama.com/library/deepseek-r1:8b),
[Ministral 3B BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16),
[Ministral 8B BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16),
[instruções vLLM da Mistral](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512).

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
python -m pip install -r requirements-validation.txt
python prepare_datasets.py --datasets race
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

## 2. Analisar e compartilhar a etapa parcial

Os resultados imutáveis desta etapa ficam em
`data/results/reflection_top1/<id>/self_eval/`. O recibo é `self_eval_receipt.json`;
isso **não** é um experimento com professor concluído.

Abra `validation_accuracy_by_similarity.ipynb`: `EXPERIMENT_ID=None` seleciona
a validação ativa; `PHASE="auto"` usa o resultado final, se disponível, ou o parcial.
`PHASE="self"` sempre abre o parcial preservado. O booleano de fallback, os
filtros, a grade, o mínimo de amostras e a cobertura ficam no início.

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
python -m pip install -r requirements-analysis.txt
python validation_ops.py analyze
```

Para exportar com fallback True, usando o ID exibido por `status`:

```bash
python analyze_validation.py --experiment-id <ID> --fallback
```

## 3. Opcional: Petrobras gera somente reflexões externas

Só faça esta etapa quando decidir acrescentar o professor. No ambiente Python
do Petrobras, com as credenciais Azure já configuradas:

```bash
git fetch origin
git switch lean-backends
git pull --ff-only origin lean-backends
python validation_ops.py restore prepare
python -m pip install -r requirements-azure.txt
python validation_ops.py start teacher
```

GPT-5.4 recebe somente questões/tentativas de **treino** dos estudantes e o
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

O comando importa o professor, gera somente teacher_simple e teacher_complex
e reaproveita exatamente as linhas de baseline/self já finalizadas. Ao terminar,
recria a análise com as cinco condições. O snapshot `self_eval/` é preservado.
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

- Falha durante a preparação: corrija o problema indicado e repita `start local`
  com a mesma configuração. As chamadas concluídas são recuperadas do cache.
- Prepare concluído, avaliação local interrompida: `python validation_ops.py start self-eval --gpu 3`.
- Professor interrompido: repita `python validation_ops.py start teacher`.
- Conclusão externa interrompida: repita `python validation_ops.py start finish --gpu 3`.
- Use `--experiment-id <ID>` com status/share/restore/analyze ou com start
  self-eval/teacher/finish para selecionar explicitamente uma run e evitar depender
  do ponteiro mais recente. Não use `--fresh` para a extensão opcional.
- Um lock local impede dois jobs simultâneos no mesmo checkout. Ele não coordena
  checkouts diferentes. Uma execução interrompida pode exigir esperar o worker
  encerrar antes de retomá-la.
- Os parâmetros científicos e fingerprints dos dados são congelados no manifesto.
  As versões de vLLM, torch, Transformers e mistral-common também entram na
  identidade da nova run e são conferidas nas etapas GPU seguintes.
  As etapas seguintes restauram os limites originais. Não misture ambientes ou
  revisões de código entre fases sem verificar compatibilidade.

Esta implementação foi validada localmente com backends simulados; a verificação
real de carga e geração dos nove modelos ocorre no preflight do servidor GPU.
