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
respeitar o formato, uma segunda geração julga apenas qual opção foi escolhida.
Respostas ainda ambíguas ficam com `correct=null` e são contabilizadas como não
resolvidas, sem serem silenciosamente convertidas em erro ou acerto.

O campo `think: false` é enviado ao Ollama. Além disso, `Generation` remove
qualquer bloco `<think>` embutido antes que a saída seja salva, avaliada ou
usada para gerar reflexão.

Os limites iniciais e da única repetição são:

| Modelo | Resposta treino | Resposta validação | Reflexão simples | Reflexão complexa |
|---|---:|---:|---:|---:|
| Phi-2 | 512 → 768 | 384 → 576 | 512 → 768 | 768 → 1152 |
| DeepSeek/Llama | 768 → 1152 | 512 → 768 | 768 → 1152 | 1024 → 1536 |

Uma seta representa a repetição seletiva feita somente quando a primeira saída
fica vazia ou termina com `finish_reason=length`. Se a segunda tentativa ainda
truncar, seu texto é apagado, o item recebe `length_exhausted` e todas as
condições dependentes ficam não resolvidas. O lote continua e a cobertura final
torna a exclusão visível.

Se a segunda tentativa continuar vazia, o tratamento é idêntico, com status
`empty_exhausted`. Isso se aplica a respostas, reflexões e ao classificador
auxiliar (`judge`).

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
