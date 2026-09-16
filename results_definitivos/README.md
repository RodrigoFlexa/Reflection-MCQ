# results_definitivos

Os resultados que valem. Um caminho fixo por split, sem id de run no meio:

```
results_definitivos/
  validation/
    all_outcomes.jsonl    linhas canônicas, uma por medida
    manifest.json         de onde veio cada linha e o que foi normalizado
    coverage.csv          aluno x condição x professor x dataset
    gaps.json             o que a grade final pede e ainda não está aqui
  test/
    (mesmos quatro arquivos)
```

Nada aqui é editado à mão. Os quatro arquivos de um split saem de um comando:

```bash
python consolidate_results.py build  --split validation   # refaz a partir dos runs brutos
python consolidate_results.py audit  --split validation   # relê o canônico e refaz cobertura/lacunas
python consolidate_results.py status                      # resumo dos dois splits
```

As fontes estão declaradas em `consolidate_results.py`, em código, na ordem de
autoridade: a primeira fonte que traz uma medida vence e as repetições depois
dela são contadas como duplicata. Rodar `build` duas vezes dá o mesmo arquivo.

## A grade final

Definida em [`rmcq/grid.py`](../rmcq/grid.py), que é de onde o pipeline, a
auditoria e os notebooks tiram a lista — não há segunda cópia.

**Alunos (9):** phi2, deepseek-r1-8b, deepseek-r1-1.5b,
llama3.1-8b, llama3.2-3b, qwen2.5-3b, qwen2.5-7b, ministral-3-3b, ministral-3-8b.

**Professores (5):**

| Professor | Ensina | Por quê |
|---|---|---|
| gpt-5-4-petrobras | os 9 alunos | externo à faixa de tamanho, é a referência |
| deepseek-r1-8b | os 5 alunos de até 3B | professor aberto |
| llama3.1-8b | os 5 alunos de até 3B | professor aberto |
| qwen2.5-7b | os 5 alunos de até 3B | professor aberto |
| ministral-3-8b | os 5 alunos de até 3B | professor aberto |

Os professores abertos não ensinam modelos do próprio porte: um 8B ensinando um
8B não é professor, é par, e o contraste que o experimento quer medir some. Os
alunos de até 3B são phi2, deepseek-r1-1.5b, llama3.2-3b,
qwen2.5-3b e ministral-3-3b.

São 29 pares professor-aluno e 85 células (aluno, condição, professor) por item.

## Identidade de uma linha

```
(model, dataset, val_uid, condition, teacher_model)
```

`teacher_model` é nulo em `baseline`, `self_simple` e `self_complex`, e nomeia o
professor nas duas condições `teacher_*`. Linhas de runs antigos não traziam o
campo porque só havia um professor possível; na consolidação elas recebem
`gpt-5-4-petrobras`, que é o que elas afirmavam implicitamente.

Cada linha também carrega `source_run` e `pipeline_version`, para que dê para
saber de qual rodada e sob qual revisão de prompt aquele número saiu sem ter de
cruzar com outro arquivo.

### O DeepSeek-R1 é padronizado na destilação sobre Llama

`deepseek-r1-8b` é `DeepSeek-R1-Distill-Llama-8B`. Onde há escolha de base, é
essa; o nome canônico esconde a destilação porque ela é constante, não porque
seja indiferente. Manter a base fixa evita atribuir a "um DeepSeek de 8B" um
comportamento que vem da base de destilação.

**O 1.5B é exceção forçada.** Não existe `DeepSeek-R1-Distill-Llama-1.5B`: a
família oficial é Qwen 1.5B/7B/14B/32B e Llama 8B/70B — a Llama não tem nada
entre 8B e 70B (índice do Hugging Face, 13/09/2026). Então `deepseek-r1-1.5b`
fica sobre Qwen por falta de alternativa.

**`DeepSeek-R1-0528-Qwen3-8B` está fora da grade**, em
`rmcq.grid.RETIRED_STUDENTS`: é outra base. A consolidação recusa as linhas
dele, que de outro modo entrariam como o modelo que não são, ou como um décimo
aluno. A chave continua registrada em `rmcq.config.MODELS` porque o run
`bf7731c4ac1f` afirma no manifest ter rodado esse checkpoint e é dele que saem
os outros oito alunos — apagá-la tornaria aquele run impossível de reabrir.

Nomes históricos do mesmo checkpoint (`deepseek-r1-distill-llama-8b`) contam
normalmente, via `rmcq.grid.MODEL_ALIASES`; `model_checkpoint`, em cada linha,
preserva de qual deles o número saiu.

## O que está faltando hoje

`gaps.json` responde isso a qualquer momento, separando duas coisas que soam
iguais: **célula sem nenhuma linha** (trabalho inteiro) e **célula parcial**
(retomada). Na validação, em 13/09/2026:

- **Oito dos nove alunos estão completos** em baseline, self_simple e
  self_complex: 22.461 linhas cada, cinco datasets, uma só revisão (v5).
- **`deepseek-r1-8b` precisa ser gerado do zero**: 22.461 linhas de
  baseline + autorreflexão. O run v5 rodou `DeepSeek-R1-0528-Qwen3-8B` nesse
  lugar, que é outra base e não conta.
- **Nenhuma condição de professor foi gerada.** 61 células ao todo, 456.707
  linhas.

Duas fontes ficaram de fora de propósito, e o motivo de cada uma está em
`manifest.json`, campo `excluded_on_purpose`: as linhas do 0528 (base errada) e
o run v4 `91ccab5e5028` inteiro. O v4 é a única destilação-Llama de 8B que
existe em disco hoje, mas cobre quatro datasets (sem RACE) e usa os prompts
antigos — entraria como revisão misturada, não como dado faltante.

## Rodar o que falta

O pipeline lê `gaps.json` e só gera o que não está lá:

```bash
python validation_ops.py start local --p1 --gpu 0    # numa GPU
python validation_ops.py start local --p2 --gpu 1    # na outra
python validation_ops.py merge                       # funde as partições
python consolidate_results.py build --split validation
```

Repetir o comando depois de uma interrupção retoma de onde parou; o que já está
no arquivo canônico não é gerado de novo.
