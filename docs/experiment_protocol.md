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

Saída vazia ou `finish_reason=length` interrompe a etapa com erro. O checkpoint
parcial é preservado para diagnóstico, mas nunca é reutilizado como reflexão
válida; isso impede que truncamentos entrem silenciosamente na avaliação.

Para modelos locais, o pipeline também mede o prompt renderizado antes de
gerar. Se a questão recuperada + reflexão + validação não couberem no contexto,
a execução falha explicitamente em vez de truncar a questão recuperada. Isso é
especialmente importante para o contexto nativo de 2048 tokens do Phi-2.
