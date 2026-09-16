# Protocolo top-1 v5: teste e auditoria

## Escopo

A avaliação usa explicitamente `test` ou `validation`; o default novo é `test`.
Os dados e limites devem ser escolhidos em validação antes da avaliação final.
O manifesto distingue os splits e inclui hashes dos arquivos de treino e
avaliação, modelos, prompts, parâmetros de template, limites e perfil.

Estudantes: baseline, self_simple, self_complex, teacher_simple,
teacher_complex. GPT-5.4: baseline, self_simple, self_complex.
Nenhuma condição adicional foi introduzida.

## Grade de professores (etapa final, 12/09/2026)

A lista canônica está em [`rmcq/grid.py`](../rmcq/grid.py); este documento
explica a decisão, não a repete.

Nove alunos: phi2, deepseek-r1-8b, deepseek-r1-1.5b,
llama3.1-8b, llama3.2-3b, qwen2.5-3b, qwen2.5-7b, ministral-3-3b, ministral-3-8b.

Cinco professores, com alcances diferentes:

- **gpt-5-4-petrobras** ensina os nove. É externo à faixa de tamanho dos alunos,
  então serve de referência contra a qual os demais são lidos.
- **deepseek-r1-8b, llama3.1-8b, qwen2.5-7b e ministral-3-8b**
  ensinam apenas os cinco alunos de até 3B (phi2,
  deepseek-r1-1.5b, llama3.2-3b, qwen2.5-3b, ministral-3-3b).

Um professor do mesmo porte do aluno não é professor, é par: um llama3.1-8b
instruindo um deepseek-r1-8b mede transferência entre iguais, que é outra
pergunta. Por isso os quatro professores abertos ficam restritos aos alunos
menores, e os quatro modelos da faixa 7B-8B aparecem como alunos apenas nas
condições sem professor e na do professor externo.

São 29 pares professor-aluno e 85 células (aluno, condição, professor) por item
de avaliação.

O DeepSeek-R1 é padronizado na destilação sobre Llama: `deepseek-r1-8b` é
`DeepSeek-R1-Distill-Llama-8B`. Manter a base constante evita atribuir a um
"DeepSeek de 8B" um comportamento que vem da base de destilação e não do
modelo. `deepseek-r1-1.5b` é a exceção forçada — `Distill-Llama-1.5B` não
existe; a família Llama tem 8B e 70B —, e por isso fica sobre Qwen.

`DeepSeek-R1-0528-Qwen3-8B` está fora da grade por ser outra base. O mapa de
nomes está em `rmcq.grid.MODEL_ALIASES` e vale para rótulo; para carregar
pesos, cada chave histórica continua apontando para o seu checkpoint, e a linha
consolidada guarda o dela em `model_checkpoint`.

Uma linha de resultado passa a ser identificada por
`(model, dataset, val_uid, condition, teacher_model)`. Sem `teacher_model` na
chave, as cinco reflexões de professor sobre a mesma questão colidiriam sob o
nome `teacher_simple`. Nas análises o eixo correspondente é o **braço**:
`teacher_simple@llama3.1-8b`, e `baseline`/`self_*` sem sufixo.

## Recuperação

Um top-1 por questão no treino completo do mesmo dataset. Fontes repetidas são
permitidas: a resposta e as duas reflexões de cada fonte são calculadas uma vez
por modelo. A recuperação não usa gabaritos nem justificativas. Os empates usam
a primeira ocorrência na ordem congelada do treino.

Duplicatas de enunciado são removidas; no RACE, a identidade para deduplicação
inclui artigo, pergunta e alternativas. Todas as questões têm UID único. O mesmo
artigo pode sustentar várias perguntas dentro de um split, sem fundi-las. Artigos
compartilhados entre treino e avaliação do RACE são removidos do treino para
impedir recuperação de outras perguntas sobre a mesma passagem de avaliação.

A preparação RACE usa `ehovy/race`, configuração `all`, preservando train,
validation e test oficiais e a divisão middle/high. A revisão resolvida e os
hashes de saída ficam em `data/processed/race/dataset_manifest.json`.

O embedder permanece BGE-large-en-v1.5. São registrados os comprimentos de
entrada antes do limite do embedder e `embedding_truncated` por par. Não foi
introduzido chunking de artigos; essa mudança seria outro protocolo de recuperação.

## Geração e integridade

No perfil final, os tetos de resposta/reflexão são 512/768 para Phi-2,
4096/4096 para DeepSeek e 1024/1024 para os demais estudantes. A profundidade
simples/complexa muda o prompt, com o mesmo teto de reflexão. O judge local
fixo usa 512 tokens (128 se configurado como Phi-2). O Azure registra seu limite
efetivo, 4000 por padrão, e o esforço de raciocínio.

Não há repetição de geração por comprimento/vazio no perfil final. O Phi-2
pode reduzir o teto de reflexão ao espaço restante no contexto, em grupos de
64 tokens quando necessário. A redução recebe `budget_reduced_for_context`.
Respostas de treino/teste têm reserva fixa. Prompts incompatíveis são marcados
individualmente, sem truncar o texto e sem interromper os outros itens.

Blocos think são removidos da saída usada; o texto bruto retornado pelo backend
local fica em `raw_text`. Um think parcial não vira reflexão utilizável.
Qwen3-8B recebe explicitamente `enable_thinking=False`; o DeepSeek continua
consumindo orçamento no seu raciocínio. O campo `completion_tokens` conta a
geração inteira. A API não fornece o seu raciocínio privado como texto bruto.

O texto completo recebido fica preservado mesmo em comprimento esgotado.
`text` é esvaziado nesses casos para evitar uso acidental na avaliação/memória;
`raw_text` e `visible_text` permitem auditoria. Reflexões completas não são
filtradas por correção, qualidade ou suspeita de alucinação. Não existe um
classificador automático de alucinação neste protocolo.

O juiz e as regras de interpretação da resposta permanecem os mesmos do
experimento anterior. Fallbacks do juiz são identificáveis por `eval_method`.

## Artefatos e marcações

Os pares mantêm `val_uid` e `validation_item` como aliases internos históricos;
a identidade real é explícita em `eval_uid`, `eval_split` e no `split` do item.
Não se deve inferir o split pelo nome do alias.

- `students/<modelo>/train.jsonl`: respostas, reflexões e seus registros brutos
  (`answer_generation`, `reflection_generations`).
- `teacher/train.jsonl` e `teacher/student_reflections/<modelo>.jsonl`: os
  mesmos registros para GPT e reflexões externas.
- `teacher/test.jsonl` ou `teacher/validation.jsonl`: avaliação do GPT.
- `analysis/all_outcomes.jsonl`: grade completa com `audit_flags`, registros da
  geração de avaliação e metadados das fontes/reflexões utilizadas.

Motivos incluem length_exhausted, empty_exhausted, prompt_context_exceeded,
transfer_context_exceeded, content_filter e falhas de fonte dependente. Flags
agregam esses motivos sem apagar o estágio ou motivo específico. Também existem
thinking_removed, partial_think, budget_reduced_for_context e embedding_truncated.

Baseline não herda falhas de uma memória que não recebeu. As condições self e
teacher herdam somente os registros da fonte e da reflexão correspondentes.

## Análise

`accuracy.csv` apresenta `n`, `resolved`, `coverage`, `accuracy` (acertos/resolvidas)
e `accuracy_all` (acertos/total). Os não resolvidos mantêm `correct=null` no dado
original, inclusive quando participam do denominador de accuracy_all.

`rmcq.analysis.filter_outcomes` é compartilhado pelo script de análise e pelo
notebook. Filtra flags, motivos, condições e níveis RACE. A opção paired usa a
interseção de questões entre as condições selecionadas, dentro de cada modelo e
dataset; não é uma interseção automática entre modelos diferentes. As seleções
salvam o total antes/depois, a cobertura e a configuração. Quantis de similaridade
são definidos antes dos filtros e não mudam entre visualizações.

Filtrar gráficos não permite avaliar retroativamente uma reflexão que não foi
utilizada. Usar saídas incompletas ou recortar memórias exige outra variante de
geração, como o notebook high_budget. Ele continua separado do protocolo final.

## Execuções históricas

A mudança de prompts e política é versionada como v5. Gerações de uma run v4
não são retomadas com o código novo; status e análise continuam disponíveis.
`--generation-profile legacy` conserva os tetos/repetições antigos em uma nova
run. O protocolo v4 original está em `_archive/docs/experiment_protocol_v4.md`.
