# Ajuste mínimo das instruções do teacher

Base revisada: `experiment_exchange/91ccab5e5028/teacher/student_reflections/`,
comparada às respostas em `students/<modelo>/train.jsonl`.

## Evidências observadas

Contagem por palavras separadas por espaço, nas reflexões externas não vazias:

| Profundidade | Reflexões | Acima da faixa solicitada | Percentual |
|---|---:|---:|---:|
| Simples, 60–120 palavras | 6.604 | 3.623 | 54,9% |
| Complexa, 160–240 palavras | 6.616 | 4.610 | 69,7% |

As medianas simples por estudante foram 121–122 palavras; as complexas,
245–250. Isso mostra descumprimento frequente da faixa, não necessariamente
truncamento pelo teto de geração.

No caso `aqua-train-033108` do Phi-2, a resposta salva tinha apenas placeholders
de raciocínio/letra, mas recebeu feedback de acerto pelo fallback do juiz.
O teacher atribuiu a ela uma estratégia de conversão de unidades que não estava
escrita. O prompt novo não corrige o juiz; evita inventar uma justificativa para
um feedback cuja resposta não fornece evidência suficiente.

Em `aqua-train-078797` do Llama, a alternativa estava marcada como correta,
mas a expressão algébrica escrita era problemática. A reflexão tratou o feedback
como apoio à interpretação geral. O diagnóstico deve separar acerto da alternativa
e validade dos passos apresentados.

## Mudanças aplicadas

1. Apoiar o diagnóstico na resposta escrita; reconhecer a falta de raciocínio
   explícito em vez de inventar passos ou intenções.
2. Não inferir raciocínio válido apenas do feedback de alternativa correta.
3. Reforçar uma ação/regra concreta e transferível, já exigida do estudante.
4. Reforçar a faixa de palavras já existente e evitar repetição.

Foram mantidos: estrutura das entradas, tópicos de análise, profundidades,
faixas de tamanho e proibição de resolver novamente ou revelar a alternativa.
O teacher continua recebendo feedback de acerto/erro, sem acrescentar o gabarito
privado recebido pelo estudante. Os prompts do estudante não foram alterados.

Essas mudanças são instruções, não uma melhoria de desempenho já medida. Seus
efeitos serão observados na nova run; o hash do manifesto impede misturar as
reflexões antigas e novas como se tivessem o mesmo prompt.
