# Reflection-MCQ

Experimento de transferência de reflexões em questões de múltipla escolha.
Cada questão de validação recupera **uma única questão de treino: a top-1 por
similaridade**. O caso recuperado contém enunciado, resposta correta, resposta
anterior do agente, resultado e reflexão. As alternativas antigas não são
repetidas.

O pipeline usa três estudantes pequenos:

- `phi2`;
- `deepseek-r1-distill-llama-8b-ollama`;
- `llama3.1-8b`.

O `gpt-5-4-petrobras` tem dois papéis: quarto estudante (responde o treino,
cria autorreflexões e responde a validação) e professor (cria reflexões para
cada resposta de treino dos três modelos pequenos).

## Condições avaliadas

Para cada modelo pequeno:

1. `baseline`: questão de validação sem memória;
2. `self_simple` e `self_complex`: reflexão criada pelo próprio estudante;
3. `teacher_simple` e `teacher_complex`: reflexão criada pelo GPT-5-4 sobre a
   resposta daquele estudante.

Para o GPT-5-4: `baseline`, `self_simple` e `self_complex`.

Não há mais grade de similaridades, placebo, threshold ou divisão
calibração/teste. Todos os itens vêm exclusivamente de `validation.jsonl` e
recebem o vizinho de treino com maior similaridade.

## Execução em dois servidores

O único entrypoint é `run_experiment.py`. Ele grava checkpoints locais em
`data/results/reflection_top1/` e os artefatos transportáveis pelo Git em
`experiment_exchange/`.

### 1. Servidor com GPU

```bash
python -u run_experiment.py prepare --gpu 3
```

O comando imprime o `experiment_id`. Faça commit e push da pasta indicada:

```bash
git add experiment_exchange/<experiment_id>
git commit -m "data: prepare reflection experiment <experiment_id>"
git push
```

Essa etapa calcula os pares top-1, faz os três estudantes responderem somente
as questões de treino selecionadas e gera suas autorreflexões simples e
complexas.

### 2. Servidor Petrobras

Depois de `git pull`, configure as credenciais Azure no `.env` e execute:

```bash
python -u run_experiment.py teacher --experiment-id <experiment_id>
```

O GPT-5-4:

- responde as questões de treino selecionadas e cria suas autorreflexões;
- cria reflexões simples/complexas de professor para cada estudante pequeno;
- responde toda a validação sem memória e com suas duas autorreflexões.

Envie os novos artefatos de volta:

```bash
git add experiment_exchange/<experiment_id>/teacher \
        experiment_exchange/<experiment_id>/teacher_receipt.json
git commit -m "data: add Petrobras stage <experiment_id>"
git push
```

### 3. Servidor com GPU novamente

```bash
git pull
python -u run_experiment.py finish --experiment-id <experiment_id> --gpu 3
```

Os resultados finais ficam em:

```text
data/results/reflection_top1/<experiment_id>/analysis/accuracy.csv
data/results/reflection_top1/<experiment_id>/analysis/all_outcomes.jsonl
```

Para conferir o andamento em qualquer máquina:

```bash
python run_experiment.py status --experiment-id <experiment_id>
```

Um ensaio rápido, sem GPU/API e sem representar o experimento real:

```bash
python run_experiment.py prepare --backend stub --models phi2 \
  --datasets arc --validation-cap 5 --train-cap 20 --embedding-device cpu
```

`--train-cap` existe apenas para smoke tests. Não o use na rodada de produção,
pois ela precisa procurar o top-1 no conjunto de treino completo.

## Prompts

Os textos canônicos ficam em `rmcq/prompts.py`:

- `build_answer_prompt` para respostas;
- `build_reflection_prompt(..., perspective="student")` para autorreflexão;
- `build_reflection_prompt(..., perspective="teacher")` para o professor;
- `build_transfer_prompt` para validação com o par top-1.

O prompt de transferência inclui, obrigatoriamente, enunciado recuperado,
resposta correta, resposta anterior, resultado e reflexão. A resposta correta
serve para interpretar o caso anterior; a instrução deixa explícito que sua
conclusão e suas letras não se transferem para a questão nova.

## Thinking

O pipeline usa duas defesas complementares:

1. `RMCQ_OLLAMA_THINK=0` envia `think: false` ao Ollama. Em versões atuais, o
   raciocínio é separado de `message.content` e pode ser desativado.
2. Todo backend remove blocos `<think>...</think>` antes de persistir ou
   reutilizar a resposta. Um bloco aberto e truncado vira resposta vazia, nunca
   uma reflexão aparentemente válida.

No Azure, modelos da família GPT-5 mantêm raciocínio interno; use
`RMCQ_AZURE_REASONING_EFFORT=low`. Esse raciocínio não aparece no conteúdo
salvo, mas ainda consome tokens do orçamento.

### Bloqueios do filtro de conteúdo do Azure

`RMCQ_AZURE_CONTINUE_ON_CONTENT_FILTER=1` impede que um único
`ResponsibleAIPolicyViolation` derrube toda a grade. Somente erros reconhecidos
como política de conteúdo são convertidos em item não resolvido; erros de
credencial, TLS, deployment ou rede continuam falhando normalmente.

Uma resposta ou reflexão filtrada nunca é reutilizada como memória. As
condições dependentes ficam com `correct=null` e um `eval_method` explícito,
como `source_answer_content_filter` ou `source_reflection_content_filter`.
`accuracy.csv` apresenta também a cobertura (`resolved / n`) e a lista detalhada
fica em `analysis/content_filter_audit.jsonl`.

## Instalação e configuração

```bash
pip install -r requirements.txt
cp .env.example .env
ollama pull deepseek-r1:8b-llama-distill-fp16
```

Aceite a licença do Llama 3.1 no Hugging Face e configure `HF_TOKEN`. No
servidor Petrobras, configure `AZURE_OPENAI_BASE_URL` (ou
`AZURE_OPENAI_ENDPOINT`) e `AZURE_OPENAI_API_KEY`. Nunca faça commit do `.env`.

Os backends disponíveis são vLLM, Transformers, Ollama, Azure OpenAI e `stub`.
Todos implementam o mesmo contrato em `rmcq/backends/base.py`.
