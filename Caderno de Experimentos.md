

**Caderno de Experimentos do paper Reflection**

Planejamento e acompanhamento do Journal paper de Reflection

Rodrigo Flexa, UNICAMP

*Última atualização: 04/08/2026*

# **Sumário**

Esse documento tem como objetivo organizar os experimentos dos artigos em desenvolvimento, cada artigo terá uma seção própria. 

O que ainda não foi decidido aparece como "A DEFINIR" ou "\[A PREENCHER\]", então iremos usar esses termos para prrocurar conteúdos. 

# **Extensão paper Aissa**

*Estenderemos "Leveraging LLM Reflection to Improve Small Language Model Agents' Capabilities"* 

## **1\. O artigo e objetivos dos experimentos** 

*\# 1\. 1 Resumo do que foi feito no paper passado*  
Esse artigo é uma extensão do trabalho apresentado no AGENTICS 2025, no qual foi feito um estudo em que SLMs podem refletir sobre os próprios erros e também podem delegar essa reflexão a um modelo externo (funcionou melhor). No experimento do paper do AGENTICS o SLM respondia, o ambiente dizia se tinha acertado, um LLM escrevia a reflexão e essa reflexão voltava para o prompt na tentativa seguinte. O ganho apareceu na acurácia, mas o resultado que mais se sustentou foi a queda nos **error loops**, ou seja, o modelo parava de insistir na mesma resposta errada.

*\# 1\. 2 Resumo das Limitações do paper (agentics)*

Dos experimentos deste paper podemos pontuar algumas limitações. A primeira é que a reflexão era reaplicada na **mesma questão** que a tinha gerado, o que mede memorização e não generalização. A segunda é de escopo: um único dataset (ARC), dois modelos respondedores e o ChatGPT-3.5 como único refletor, o que não permite dizer se o efeito vale fora daquele arranjo específico.

*\# 1\. 3 O que pretendemos com essa extensão*

Este trabalho amplia os experimentos em três frentes: separa treino e teste para recuperar reflexões por similaridade semântica (entre questões), expande a avaliação para cinco benchmarks de diferentes tipos de raciocínio e compara todos os pares viáveis de professor e aluno entre quatro modelos, considerando reflexões simples e complexas. O objetivo é analisar o impacto da reflexão externa e da autorreflexão em diferentes tarefas, medindo ganhos de desempenho, redução de alucinações, custo computacional e influência do tipo de problema, da origem da reflexão e da similaridade entre questões, com a expectativa de maiores benefícios em tarefas de matemática e lógica do que em benchmarks dependentes de conhecimento factual.

**Destino.** Ainda não decidido.

**Onde mora.** Este repositório. Setup em `scripts/`, preparação de dados em `notebooks/01`, teste de inferência em `notebooks/02`.

## **2\. Dados e modelos**

Cinco datasets de múltipla escolha. O ARC vem do paper anterior e os outros quatro são novos.

O tamanho do conjunto de treino de cada dataset foi definido pela **fórmula de Cochran** com 95% de confiança, margem de 5 pontos, p \= 0,5 e correção para população finita. A amostragem é estratificada pelo gabarito e usa semente 42\. Ver `notebooks/01_formatacao_e_selecao.ipynb`.

O treino é **deduplicado antes da amostragem**, contra validação e teste e contra si mesmo, pela chave (contexto \+ enunciado) normalizada. Sem isso, 3,6% do treino do LogiQA2 reaparece nos splots de avaliação e 40% das questões de teste do LogiQA2 têm par idêntico no treino: a recuperação por similaridade devolveria a reflexão da questão idêntica, com cosseno 1,0, reproduzindo o confundidor de memorização descrito na seção 1.2. O custo da deduplicação é de 3 questões no total, porque com p \= 0,5 o n de Cochran é quase insensível a N.

Os tamanhos abaixo são os que os scripts produzem hoje, verificados. Substituem os valores anteriores (708 ARC, 3.806 OpenBookQA, 5.611 GSM8K, 9.360 LogiQA2, 5.244 AQuA), que não vinham de Cochran e implicariam cerca de 31 mil questões de treino por aluno.

| Dataset | Tipo de Problema | População (pós-dedup) | Treino (Cochran) | Validação | Teste |
| :---- | :---- | :---- | :---- | :---- | :---- |
| gsm8k | processo, aritmética multi-passo | 7.473 | 366 | — | 1.319 |
| aqua | processo, álgebra e quantitativo | 76.992 | 383 | 252 | 246 |
| logiqa2 | processo, dedução lógica | 11.168 | 372 | 1.565 | 1.565 |
| arc | conhecimento, ciências escolares | 1.109 | 286 | 298 | 1.171 |
| openbookqa | conhecimento, ciências e fato | 4.790 | 356 | 500 | 500 |
| **total** |  |  | **1.763** | **2.615** | **4.801** |

Contagens de teste abaixo do oficial porque itens com duas alternativas de texto idêntico são descartados: o gabarito deles é ambíguo. São 8 no AQuA, 7 no LogiQA2 e 1 no ARC. No treino do AQuA isso remove 2.528 itens.

\# Nota sobre o GSM8K: é o único dataset que **não é múltipla escolha na origem**. A conversão usa os resultados intermediários que o próprio dataset anota em `<<expr=valor>>` como distratores, porque é onde um modelo que erra o número de passos costuma parar; distratores aleatórios seriam eliminados por estimativa de ordem de grandeza. Consequência a declarar no paper: **os números de GSM8K aqui não são comparáveis com GSM8K aberto da literatura.**

Quatro modelos abertos estão registrados, todos aptos aos dois papéis. **A rodada atual usa dois**, controlado por `RMCQ_ACTIVE_MODELS` no `.env`; os outros ficam no registro, com repo\_id e licença, prontos para voltar sem reescrever nada.

| Modelo | Parâmetros | repo\_id | Rodada atual | Observação |
| :---- | :---- | :---- | :---- | :---- |
| Phi-4 Mini | 3.8B | `microsoft/Phi-4-mini-instruct` | **ativo** | MIT, exige `trust_remote_code` |
| Llama-3-8B-Instruct | 8B | `meta-llama/Meta-Llama-3-8B-Instruct` | **ativo** | gated com aprovação manual, exige HF\_TOKEN |
| Qwen3-8B | 8B | `Qwen/Qwen3-8B` | fora | Apache 2.0, pensamento híbrido, exige transformers ≥ 4.51 |
| Mistral-7B-Instruct-v0.3 | 7B | `mistralai/Mistral-7B-Instruct-v0.3` | fora | Apache 2.0 |

O par Phi-4 Mini e Llama-3-8B dá 4 combinações, e o desenho mínimo já contém as duas condições centrais: a diagonal é autorreflexão, e fora dela ficam as **duas direções** da reflexão externa. Isso permite testar de saída uma pergunta que o paper anterior não podia fazer, porque só tinha o ChatGPT-3.5 como professor: **um modelo de 3.8B consegue escrever reflexão útil para um de 8B, ou o benefício da reflexão externa depende de o professor ser mais capaz que o aluno?** A assimetria entre os dois pares fora da diagonal responde isso diretamente.

\# Notas:

Este conjunto muda o que estava planejado antes e vale registrar o que se perde. O par Llama e DeepSeek-R1-Distill era justificado porque, sendo o segundo destilado do primeiro, a diferença entre os dois isolava o efeito do treino de raciocínio com o resto da arquitetura praticamente constante. **Sem o DeepSeek não há par controlado**, e nenhuma comparação entre os quatro modelos atuais separa arquitetura de treino de raciocínio. O Qwen3 recupera parte disso, porque ligar e desligar `enable_thinking` no mesmo modelo é um contraste ainda mais limpo, mas é um eixo diferente. Também saem o Phi-2 e o ChatGPT-3.5, que faziam a ponte com os resultados do AGENTICS: sem eles, a comparação com o paper anterior passa a ser indireta.

Sobre alguns parâmtros: 

* Aluno: temperatura 0\.  
* Professor: temperatura 0.8.  
* Máximo: 4096 tokens.  
* Decodificação: greedy \+ seed 42\.

## **3\. Os experimentos**

***\#  Criação do baseline***  
Antes de qualquer coisa rodamos o baseline, no qual os modelos respondem tudo (treino e teste) sem nenhuma reflexão. É o baseline contra o qual todo o resto é comparado.

***\# Etapa de Treino***  
Na etapa de treino cada aluno responde às questões do split de treino (reaproveita o baseline) com raciocínio passo a passo, e o professor escreve uma reflexão para cada resposta, tenha ela sido certa ou errada. 

***\# Etapa de Avaliação***  
Na etapa de avaliação o aluno responde questões do split de teste com as reflexões em que as questões são  semanticamente mais próximas,  injetado-as no prompt. 

***\# Detalhes de Output***

No repositório existe um único prompt (detalhado abaixo), um único extrator e um único formato de saída, todos em `rmcq/common.py`. Se formos colocar algum tratamento específico de modelo isso vira coluna no JSONL em vez de caminho de código separado.

***\# Padronização das alternativas***

**Invariante do projeto: todo item, em todo dataset, usa rótulos de letra contíguos começando em A.** A conversão acontece no notebook 01, no momento em que os arquivos de `data/splits/` são escritos, e é verificada ali com `assert` — nenhum arquivo sai com rótulo numérico ou fora de ordem. Três dos cinco datasets chegam sem isso:

| Dataset | Como vem da origem | Como fica |
| :---- | :---- | :---- |
| ARC | parte dos itens com rótulos `"1".."4"` | `A, B, C, D` |
| AQuA | opções como `["A)5", "B)10", ...]` | rótulo separado do texto: `A` \+ `5` |
| LogiQA2 | `answer` como índice inteiro `0..3` | `A, B, C, D` |
| GSM8K | não era múltipla escolha | `A, B, C, D` após gerar distratores |
| OpenBookQA | já em letras | inalterado |

A tradução é sempre **por posição**, nunca pelo rótulo textual do original. Isso importa no ARC: um item que vem com `["1","2","3","4"]` e gabarito `"3"` vira gabarito `C` porque `"3"` é o terceiro rótulo, não por semelhança entre `3` e `C`.

**O número de alternativas varia, e isso é aceito.** O AQuA tem 5 em todos os itens; o ARC tem 7 itens com 3 alternativas e 4 com 5, entre 1.755\. O invariante é sobre a FORMA do rótulo, não sobre a quantidade, porque as duas partes do pipeline que dependem disso já tratam a variação item a item: **o extrator** monta o conjunto de letras aceitas a partir das alternativas do próprio item, então um item de 3 alternativas nunca aceita `D`; e **a análise** pondera a acurácia do chute por `num_choices`, item a item, em vez de assumir 25% para todo mundo. O que quebraria as duas coisas é rótulo numérico ou fora de ordem, porque aí qualquer número solto no raciocínio se tornaria candidato a resposta.

O manifesto `data/splits/manifest_splits.json` registra o esquema, a distribuição de `num_choices` por dataset e o fato de a verificação ter rodado.

***\# Como a resposta correta é detectada***

O gabarito é a **letra canônica** gravada em `answerKey` no notebook 01\. A correção é comparação exata de letra, `predicted == answerKey`. Nenhuma comparação de conteúdo, nenhuma tolerância. A origem do `answerKey` varia por dataset e em todos os casos é resolvida **por posição**, nunca pelo rótulo textual do original: ARC e OpenBookQA traduzem o rótulo original (que às vezes é `1..4`), AQuA usa o campo `correct`, LogiQA2 usa o índice inteiro `answer`, e o GSM8K usa a posição do número-gabarito no pool embaralhado.

Extrair a letra da resposta do modelo é o passo que pode falhar em silêncio, então a extração é uma **cascata de cinco regras**, e a que foi usada fica registrada em `extraction_method` em cada linha do JSONL:

| Nível | Método | O que reconhece |
| :---- | :---- | :---- |
| 1 | `strict` | `FINAL ANSWER: B`, exatamente como o prompt pede. **Só este conta como `followed_format`** |
| 2 | `loose_final` | a mesma linha com ruído: `**FINAL ANSWER:** (C)` |
| 3 | `answer_is` | resposta em prosa: "the correct answer is D" |
| 4 | `value_match` | o modelo emitiu o **valor** da alternativa, não a letra |
| 5 | `bare_letter` | uma letra sozinha, e só nas **3 últimas linhas** |
| — | `none` | abstenção. `is_correct` fica nulo, não falso |

Quatro decisões que valem registrar:

**Blocos `<think>` são removidos antes de extrair.** Sem isso, o Qwen3 e qualquer modelo de raciocínio teriam a resposta capturada do rascunho interno, que frequentemente cita uma letra diferente da conclusão.

**A última ocorrência vale**, em todas as regras. O modelo costuma enumerar alternativas antes de concluir.

**`value_match` existe por causa do GSM8K.** Lá as alternativas são números, e um modelo que acabou de fazer a conta termina naturalmente com `FINAL ANSWER: 72` em vez da letra. Sem essa regra isso conta como abstenção, e como a acurácia do paper conta abstenção como erro, a acurácia do GSM8K cairia por um motivo que não tem nada a ver com raciocínio. A regra normaliza vírgula, `$` e `%`, compara numericamente quando os dois lados são números (`0.20` casa com `0.2`), e **abstém se o valor casar com duas alternativas** — empate ali é ambiguidade real, e desempatar inventaria acurácia.

**`bare_letter` foi restrita ao fim da resposta.** Uma linha contendo só `B)` no meio do texto é quase sempre parte de uma enumeração das alternativas, não a conclusão.

A tabela "Como a resposta foi detectada" em `results/analysis/summary.md` mostra a distribuição por modelo, e `accuracy_strict_format_only` dá a acurácia restrita às linhas que seguiram o formato. Se as regras de resgate estiverem carregando muito peso para algum modelo, isso é achado sobre aderência a instrução e merece uma linha no paper — não é bug a esconder no extrator.

***\# Implementações a mais***

Além do baseline sem reflexão e da auto-reflexão, que era o baseline do paper anterior, iremos incluir mais duas condições: 

* Retry com feedback e sem reflexão. O simples "sua resposta estava incorreta" já elimina uma alternativa sozinho, então sem essa condição o ganho atribuído à reflexão fica superestimado. Talvez adicionar uma alternativa a mais fique mais justo (menos chance do modelo chutar).

* Self-consistency com orçamento de inferência equivalente, isto é, N amostras com temperatura mais votação majoritária. A literatura recente mostra self-consistency batendo multi-agent debate por 88,2 contra 83,0 no GSM8K.

***\# Resumo variáveis do experimento***

* Modelo professor (quem escreve a reflexão) e Modelo estudante  
* Reflexão entre simples e complexa,   
* O número k de reflexões recuperadas (1, 3 e 5 como valores candidatos)  
* Dataset. 

\# Escopo da primeira rodada:

| Variável | Valores | Total |
| :---- | :---- | :---- |
| Par aluno-professor | Phi-4 Mini e Llama-3-8B, nos dois papéis | 4 |
| Profundidade | simples, complexa | 2 |
| k | **3** | 1 |
| Dataset | os cinco | 5 |
| **Configurações de avaliação** |  | **8** (× 5 datasets) |

**k \= 3 e não 1 nem 5, por dosagem.** Com k \= 1 um resultado nulo é ambíguo: não dá para separar "reflexão recuperada não transfere" de "uma reflexão sobre outra questão é sinal insuficiente". Com k \= 5 o prompt cresce e a diluição entra como confundidor antes de sabermos se o efeito existe. O k \= 3 dá ao mecanismo uma chance justa de aparecer.

**A ressalva:** a figura de transferability fica mais limpa em k \= 1, porque aí a atribuição entre similaridade e utility é bivariada — uma reflexão, um cosseno, um resultado. Com k \= 3, se a utility for alta, não se sabe se veio da mais próxima ou da terceira. Cada linha guarda `top1_similarity` e `mean_similarity`, então a figura continua possível; a versão de atribuição limpa exige rodar k \= 1, o que **não regera nada** (baseline, reflexões e índice são compartilhados entre valores de k).

Custo desta rodada, medido com `--dry-run`: cerca de 6 h de GPU com vLLM, contra ~26 h da grade completa.

## **4\. Prompts**

O prompt de resposta está congelado em scripts/common.py e vale para todos os modelos, datasets e etapas. 

| You are answering a multiple-choice question. Question: {question} Options: {options} Instructions: \- Think step by step before answering. \- Choose exactly one option. \- End your response with this exact line, and nothing after it: FINAL ANSWER: \<letter\> |
| :---- |

Os prompts de reflexão formam uma grade de dois por dois, cruzando profundidade, entre simples e complexa, com perspectiva, entre aluno e professor. **A versão do professor é a versão do aluno com um preâmbulo em terceira pessoa e a troca dos pronomes, e nada mais.**Simples, na perspectiva do aluno (auto-reflexão)

| You are given: 1 \- The original multiple-choice question. 2 \- Your previous answer. 3 \- Feedback indicating whether your answer was correct or incorrect. Write a brief reflection (3-6 sentences) on your previous response. Discuss: \- The main factors that influenced your answer. \- Any assumptions or uncertainties you had. \- How the feedback supports or challenges your approach. \- One lesson you would apply when answering similar questions in the future. If the answer was correct, explain why your approach was effective and note any remaining uncertainty. If the answer was incorrect, identify the most likely source of the error without simply restating that the answer was wrong. Do not answer the question again or identify which option is correct. |
| :---- |

### **Complexa, na perspectiva do aluno (auto-reflexão)**

| You are given: 1 \- The original multiple-choice question. 2 \- Your previous answer. 3 \- Feedback indicating whether your answer was correct or incorrect. Write a detailed reflection on your previous response. Analyze: \- The reasoning strategy you used to reach your answer. \- The evidence or cues from the question that influenced your decision. \- Any assumptions, heuristics, or uncertainties that affected your judgment. \- How the feedback confirms or contradicts your reasoning. \- Whether your conclusion depended on missing knowledge, incorrect interpretation,   overconfidence, or insufficient evaluation of alternatives. \- How you would improve your reasoning process for similar problems in the future. If the answer was correct, explain which parts of your reasoning were reliable and whether your confidence was appropriately calibrated. If the answer was incorrect, explain what aspect of your reasoning should change rather than merely noting the correct outcome. Do not answer the question again, identify the correct option, or speculate about what the correct answer is. |
| :---- |

### **Simples, na perspectiva do professor (reflexão externa)**

| You are reviewing the response of another language model (the student model) to a multiple-choice question. Your task is to write a reflection about that model's response, addressed to it, so that it can do better on similar questions later. You are not the one answering the question. You are given: 1 \- The original multiple-choice question. 2 \- The student model's previous answer (with its reasoning). 3 \- Feedback indicating whether that answer was correct or incorrect. Write a brief reflection (3-6 sentences) on the student model's response. Discuss: \- The main factors that influenced its answer. \- Any assumptions or uncertainties visible in its reasoning. \- How the feedback supports or challenges its approach. \- One lesson it should apply when answering similar questions in the future. If the answer was correct, explain why its approach was effective and note any remaining uncertainty. If the answer was incorrect, identify the most likely source of the error without simply restating that the answer was wrong. Do not answer the question yourself or identify which option is correct. |
| :---- |

### **Complexa, na perspectiva do professor (reflexão externa)**

| You are reviewing the response of another language model (the student model) to a multiple-choice question. Your task is to write a reflection about that model's response, addressed to it, so that it can do better on similar questions later. You are not the one answering the question. You are given: 1 \- The original multiple-choice question. 2 \- The student model's previous answer (with its reasoning). 3 \- Feedback indicating whether that answer was correct or incorrect. Write a detailed reflection on the student model's response. Analyze: \- The reasoning strategy it used to reach its answer. \- The evidence or cues from the question that influenced its decision. \- Any assumptions, heuristics, or uncertainties that affected its judgment. \- How the feedback confirms or contradicts its reasoning. \- Whether its conclusion depended on missing knowledge, incorrect interpretation,   overconfidence, or insufficient evaluation of alternatives. \- How it should improve its reasoning process for similar problems in the future. If the answer was correct, explain which parts of its reasoning were reliable and whether its confidence was appropriately calibrated. If the answer was incorrect, explain what aspect of its reasoning should change rather than merely noting the correct outcome. Do not answer the question yourself, identify the correct option, or speculate about what the correct answer is. |
| :---- |

### **Resposta com reflexões recuperadas**

Congelado em `rmcq/common.py` (`RETRIEVED_HEADER`). O prefixo abaixo é concatenado ao prompt de resposta padrão, sem alterá-lo.

| Below are lessons you recorded after answering other, different multiple-choice questions in the past. They are ordered from least to most relevant to the question you are about to answer. They are not about the question below and they do not contain its answer. Use them only as guidance on how to reason. \[Lesson 1\] {reflexão} \[Lesson 2\] {reflexão} ... \--- |
| :---- |

**Três decisões de formato, com a razão de cada uma:**

**Ordem: similaridade crescente.** A reflexão mais parecida com a questão nova fica por **último**, imediatamente antes dela. Decodificação causal atende mais ao que está perto do fim do prompt, então essa posição dá mais peso ao conselho mais relevante. Também mantém a posição relativa da mais similar estável entre k \= 1, 3 e 5, o que é necessário para que a comparação entre valores de k isole o efeito da quantidade e não o da posição.

**A questão de origem NÃO é incluída.** Incluir transformaria a reflexão em exemplo few-shot de uma questão quase idêntica, que é exatamente o confundidor de memorização que a extensão existe para eliminar. Também inflaria o prompt (as premissas do LogiQA2 têm cerca de 100 tokens cada) em 460 mil gerações. Existe como ablação, via `RMCQ_INJECT_SOURCE_QUESTION=1`.

**A negação explícita ("They are not about the question below and they do not contain its answer") é necessária.** Sem ela, o modelo tende a ler as reflexões como feedback sobre a própria resposta à questão atual, e passa a procurar nelas a alternativa correta.

\# Nota sobre candidatos: a recuperação só considera itens de treino que **têm reflexão gerada**. Sem esse filtro, um arquivo rotulado k \= 3 pode receber uma reflexão só quando a etapa de reflexão está incompleta, e nada quebra para avisar. O número real injetado por linha fica em `extra.n_reflections_injected`.

## **5\. Métricas**

*  **Acurácia: Percentual de acerto**  
*  **Comprimento da reflexão:** Palavras e caracteres.  
*  **Similaridade com a questão:** Cosseno entre reflexão e questão original. Baixa significa reflexão abstrata, alta significa presa ao caso.  
*  **Diversidade lexical:**  TTR. palavras únicas por palavras totais  
*  **Redução de Alucinações:** Quantidade de alucinações no baseline em relação a reflexão  
*  **Reflection Utility:** Taxa de virada errado para certo menos certo para errado, no teste, com reflexão contra sem. Subtrair o segundo termo impede que uma reflexão que conserta cinco e estraga cinco apareça como ganho.   
* **Transferability**. A mesma utility condicionada à similaridade entre a questão de teste e a que gerou a reflexão. O gráfico de utility contra similaridade é a figura mais forte do paper.  
* Persistência de erro. Quanto dos erros do treino reaparece no teste apesar da reflexão em memória. Versão transferível do error loop,  
* Custo. Tokens de entrada e saída, chamadas e tempo por configuração. Sem isso não há como dizer se a reflexão complexa se paga, nem como comparar com self-consistency.

## **6\. Em que pé está**

*Status: \[ \] não iniciado, \[\~\] em andamento, \[x\] concluído, \[\!\] bloqueado.*

|  | Etapa | Prazo | Atualizado |
| :---- | :---- | :---- | :---- |
| \[x\] | Repositório do zero: `scripts/config.py`, `common.py`, setup de dados e modelos |  | 11/08/2026 |
| \[x\] | Construção dos datasets: download, schema unificado, dedup e seleção de Cochran |  | 11/08/2026 |
| \[x\] | Prompts de reflexão (congelados em `scripts/common.py`) |  | 02/08/2026 |
| \[x\] | Extrator único, com cascata de formatos e abstenção separada de erro |  | 11/08/2026 |
| \[x\] | Framework `rmcq`: CLI, backends vLLM/transformers/stub, store retomável |  | 12/08/2026 |
| \[x\] | Prompt de injeção das k reflexões (seção 4\) |  | 12/08/2026 |
| \[x\] | Modelo de embeddings escolhido: `BAAI/bge-large-en-v1.5`; k \= 3 na primeira rodada |  | 12/08/2026 |
| \[x\] | Código de todas as etapas: baseline, reflexão, índice, avaliação, retry, self-consistency, análise |  | 12/08/2026 |
| \[x\] | Pipeline validado de ponta a ponta com backend stub, sem GPU |  | 12/08/2026 |
| \[\~\] | Notebook de teste de inferência: escrito, **falta rodar em GPU** | A DEFINIR | 11/08/2026 |
| \[ \] | **Rodada 1:** baseline nos dois modelos ativos (18 mil gerações, ~0,9 h) | A DEFINIR |  |
| \[ \] | **Rodada 1:** reflexões (14 mil gerações, ~0,7 h) | A DEFINIR |  |
| \[ \] | **Rodada 1:** avaliação com k \= 3 (38 mil gerações, ~1,8 h) | A DEFINIR |  |
| \[ \] | **Rodada 1:** retry e self-consistency (~2,3 h) | A DEFINIR |  |
| \[ \] | Rodada 2: abrir k para {1, 3, 5}; k \= 1 é o que dá a figura limpa de transferability | A DEFINIR |  |
| \[ \] | Rodada 3: reincluir Qwen3-8B e Mistral-7B | A DEFINIR |  |
| \[ \] | Modelos de API da OpenAI como professor (fora do escopo por ora) | A DEFINIR |  |
| \[ \] | Submissão | A DEFINIR |  |

\# Comandos:

`python -m rmcq status` mostra o progresso de cada etapa. `python -m rmcq <etapa> --dry-run` estima gerações, tokens e horas antes de começar. Toda etapa é retomável por `uid`. Ver README.

\# Pendências abertas que o setup expôs:

* **O baseline anterior não é reaproveitável.** Estava marcado como concluído em 02/08, mas o conjunto de modelos mudou (entram Qwen3 e Mistral, saem DeepSeek, Phi-2 e ChatGPT-3.5) e os splits de treino mudaram de tamanho e de conteúdo. Precisa ser regerado.
* **Sem par controlado de modelos.** Ver a nota da seção 2\. Decidir se o DeepSeek-R1-Distill-Llama-8B volta ao desenho.
* **LogiQA 2.0 tem itens com premissa desalinhada da pergunta**, defeito do release oficial (o mirror em parquet foi conferido e é fiel ao original). Não há detector automático confiável, porque heurísticas de sobreposição lexical confundem esses itens com as questões de analogia. Com 372 itens de treino, inspeção manual é viável.
* **Falta batching.** 1.763 questões de treino mais 4.801 de teste, vezes quatro alunos, vezes as condições, não fecha uma questão por vez.

# 