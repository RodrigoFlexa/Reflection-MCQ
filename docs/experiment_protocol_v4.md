# Protocolo experimental top-1

## Unidade de recuperação

Para cada item de validação, o embedder `BAAI/bge-large-en-v1.5` calcula a
similaridade com todos os itens de treino do mesmo dataset. Apenas o maior
escore é retido. Duplicatas internas e enunciados presentes nos dois splits são
removidos antes da busca.

O conjunto de treino efetivamente respondido por cada modelo é a união desses
vizinhos top-1. Portanto, uma questão recuperada por vários itens de validação é
respondida e refletida uma única vez.

## Fluxo de artefatos

```text
GPU / prepare
  pares top-1 + respostas/reflexões dos estudantes
                       |
                       | Git
                       v
Petrobras / teacher
  respostas/reflexões próprias do GPT-5-4
  reflexões de professor para phi2/deepseek/llama
  validação do GPT-5-4 (baseline e self)
                       |
                       | Git
                       v
GPU / finish
  validação dos estudantes (baseline, self e teacher)
  tabela consolidada de acurácia
```

Cada arquivo possui `source_uid` e `val_uid`, impedindo que uma reflexão seja
associada ao par errado. O manifesto congela modelos, datasets, embedder e os
quatro prompts. O `experiment_id` é o hash dessa configuração.

## Estrutura do intercâmbio

```text
experiment_exchange/<experiment_id>/
├── manifest.json
├── retrieval_audit.json
├── pairs/<dataset>.jsonl
├── students/<model>/train.jsonl
├── prepare_receipt.json
├── teacher/train.jsonl
├── teacher/student_reflections/<model>.jsonl
├── teacher/validation.jsonl
└── teacher_receipt.json
```

Os arquivos em `pairs/` carregam os objetos completos de treino e validação.
Assim, o servidor Petrobras não depende de `data/processed`, que não é
transportado pelo Git.

Na avaliação, o caso de treino é renderizado com cinco campos: enunciado,
resposta correta, resposta anterior do agente, resultado e reflexão. A lista
de alternativas do caso antigo é omitida. As alternativas aparecem somente na
questão de validação que precisa ser respondida.

## Integridade da resposta

O parser usa a última ocorrência de `FINAL ANSWER: <letter>`. Se o modelo não
respeitar o formato, uma segunda geração julga apenas qual opção foi escolhida
— mas não é o próprio modelo que respondeu quem julga. Um `--judge-model` fixo
(padrão `llama3.1-8b`, ver `DEFAULT_JUDGE` em `run_experiment.py`) resolve o
fallback de todo mundo, incluindo suas próprias respostas. Isso evita que um
modelo que ignora instrução de formato (o DeepSeek, na prática) também seja
quem decide se acertou: ele receberia a resposta correta e uma instrução
igualmente específica ("responda só uma palavra"), que também não seguiria de
forma confiável. Respostas ainda ambíguas ficam com `correct=null` e são
contabilizadas como não resolvidas, sem serem silenciosamente convertidas em
erro ou acerto.

Isso muda a ordem de execução de `prepare`/`finish`: em vez de cada modelo
responder, julgar a si mesmo e refletir em uma única sessão, agora são três
passagens — (1) cada modelo gera suas respostas e descarrega, (2) o juiz fixo
carrega uma única vez e resolve o fallback de todos, descarrega, (3) cada
modelo recarrega para gerar suas reflexões usando o veredito já resolvido.
Custa recarregar cada modelo estudante duas vezes em vez de uma, mas isolar o
julgamento do modelo que respondeu vale o tempo de carga extra.

O campo `think: false` é enviado ao Ollama, mas nenhum modelo do registro
padrão usa mais esse provider — o DeepSeek foi movido para vLLM por
velocidade (continuous batching em vez de uma chamada HTTP por item). Isso
não muda o comportamento do `<think>`: a destilação do DeepSeek-R1 embute o
raciocínio de forma incondicional, sem alternância para desativá-lo, então
todo backend sempre abre um bloco `<think>`. A defesa real é a outra:
`Generation` remove qualquer bloco `<think>` embutido antes que a saída seja
salva, avaliada ou usada para gerar reflexão, em qualquer backend.

O Phi-2 é um modelo base atrás do wrapper de completação `Instruct:/Output:`,
sem EOS confiável: depois de terminar a resposta real, ele tende a continuar
completando no estilo do corpus de pré-treino, inventando um novo exercício
em vez de parar. Três stop sequences (`\nInstruct:`, `\nExercise`,
`\nQuestion:`) cortam a geração nesse ponto, preservando a resposta real. Elas
valem para toda geração do Phi-2 — resposta de treino, validação e reflexão —
e não se aplicam a nenhum outro modelo.

Os limites iniciais e da única repetição são:

| Modelo | Resposta treino | Resposta validação | Reflexão simples | Reflexão complexa |
|---|---:|---:|---:|---:|
| Phi-2 | 512 → 768 | 384 → 576 | 256 → 384 | 384 → 512 |
| DeepSeek/Llama | 1024 → 2048 | 512 → 768 | 768 → 1152 | 1024 → 1536 |

O `judge` não tem mais um teto por modelo respondente — é sempre o teto do
`--judge-model` fixo (`judge_budget`/`judge_retry_budget` em
`run_experiment.py`), hoje 512 → 2048 para o Llama3.1-8B padrão, 128 → 192 se
o juiz declarado for o Phi-2.

A resposta de treino do DeepSeek/Llama foi revista para 1024 → 2048 depois de
auditar `data/results`: o DeepSeek-R1-distill gasta o orçamento dentro de
`<think>...</think>`, que é removido antes de salvar — um bloco que não fecha
vira resposta vazia, não uma resposta truncada mas usável. No teto anterior
(768 → 1152), 20% das respostas de treino eram descartadas como
`length_exhausted` mesmo com a repetição, e o p99 das gerações bem-sucedidas
já estava em 1085/1152 tokens — a cauda passava do teto antigo.

Até este ponto do desenvolvimento, o `judge` era o próprio modelo que
respondeu, com um teto fixo de 128 → 256 para todo mundo, nunca ajustado por
modelo. Um piloto do DeepSeek revelou dois problemas empilhados: primeiro, ele
quase nunca produz o literal `FINAL ANSWER:` (só 4,5% das respostas
bem-sucedidas no piloto), então quase toda resposta caía no `judge`; segundo,
o `judge` sofria do mesmo problema do `<think>` que a resposta de treino
tinha — mesmo pedindo uma única palavra, o modelo ainda abria um bloco de
raciocínio antes de responder. Com o teto antigo, 40% das chamadas ao `judge`
no piloto terminaram `length_exhausted` (texto vazio), deixando ~44% dos itens
de treino sem veredito e, portanto, sem reflexão gerada.

Subir o teto do `judge` teria resolvido só metade do problema — a outra
metade (85% de fallback rate) vem do próprio DeepSeek não seguir o formato
pedido, o que não é corrigível por budget. Por isso o `judge` deixou de ser o
modelo respondente e virou um `--judge-model` fixo (padrão Llama3.1-8B, ver
acima).

Uma seta representa a repetição seletiva feita somente quando a primeira saída
fica vazia ou termina com `finish_reason=length`. Se a segunda tentativa ainda
truncar, seu texto é apagado, o item recebe `length_exhausted` e todas as
condições dependentes ficam não resolvidas. O lote continua e a cobertura final
torna a exclusão visível.

Se a segunda tentativa continuar vazia, o tratamento é idêntico, com status
`empty_exhausted`. Isso se aplica a respostas, reflexões e ao classificador
auxiliar (`judge`).

Quando o orçamento da repetição não cabe na janela do modelo, o item é
descartado antes da nova geração como `length_exhausted`, com motivo
`retry_exceeds_context`. O enunciado nunca é truncado para abrir espaço, e os
outros itens elegíveis seguem para a repetição.

Antes da validação, o pipeline mede o prompt completo. O Phi-2 rejeita qualquer
reflexão recuperada acima de 512 tokens (`reflection_token_limit_exceeded`). Se
o prompt ainda ultrapassar a janela depois de reservar 384 tokens para a
resposta, somente aquela condição recebe `transfer_context_exceeded`. Llama
3.1 e DeepSeek não precisam do teto de memória de 512 tokens, mas também passam
pela verificação contra suas janelas operacionais.

Os prompts de reflexão trazem uma faixa de frases e palavras (student simple
3–5 frases/60–100 palavras, student complex e teacher complex 8–12/160–240,
teacher simple 3–6/60–120) para reduzir divagação, especialmente no Phi-2.
Nenhum limite bruto de tokens é acrescentado ao texto; o controle de tokens
continua exclusivamente nos parâmetros de geração e na elegibilidade do
prompt de transferência.

No Azure, os defaults são `RMCQ_AZURE_MAX_TOKENS=1024` e
`RMCQ_AZURE_REASONING_MIN_TOKENS=4000`; portanto, o GPT-5-4 recebe um teto
efetivo de 4000 tokens, incluindo raciocínio interno. O parâmetro por chamada
não substitui esse teto global.

Para modelos locais, o pipeline também mede o prompt renderizado antes de
gerar. Se a questão recuperada + reflexão + validação não couberem no contexto,
a execução falha explicitamente em vez de truncar a questão recuperada. Isso é
especialmente importante para o contexto nativo de 2048 tokens do Phi-2.

Uma revisão de prompts gera outro `experiment_id`, mas pares top-1 podem ser
copiados automaticamente de uma execução anterior quando datasets, limites,
seed e modelo de embeddings forem idênticos. Essa recuperação só ocorre sem
`--fresh`.

## Temperatura

Respostas de treino, respostas de validação e julgamentos são determinísticos
(`temperature=0.0`). As reflexões dos estudantes pequenos usam
`temperature=0.7`, configurável por `--reflection-temperature`; o valor
participa do hash do checkpoint.

O GPT-5-4 Petrobras é um deployment de raciocínio e nunca recebe o parâmetro
`temperature`. Tanto suas autorreflexões quanto suas reflexões de professor usam
a configuração padrão do serviço Azure.

## Filtro de conteúdo do Azure

Bloqueios identificados como `ResponsibleAIPolicyViolation`, `content_filter`,
`jailbreak` ou conteúdo malicioso são tratados como ausência de dado. O item é
salvo com `correct=null`, a memória dependente não é usada e o restante do lote
continua. Outros erros HTTP não são engolidos.

Os recibos registram o total de eventos do filtro. Ao final,
`accuracy.csv` informa a cobertura de cada condição e
`content_filter_audit.jsonl` lista todas as condições afetadas. Assim, a
acurácia é calculada entre respostas resolvidas sem esconder quantos exemplos
foram bloqueados.
