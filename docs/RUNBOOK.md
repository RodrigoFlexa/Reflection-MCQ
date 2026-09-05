# Roteiro da run final: GPU → GitHub → Petrobras → GitHub → GPU

Execute os blocos na pasta `Reflection-MCQ`, com o ambiente Python já usado
naquele servidor ativado. Branch compartilhada: `lean-backends`.
GPU selecionada: **3**; troque `--gpu 3` nos dois comandos GPU se necessário.
Não execute duas cópias da mesma etapa em checkouts diferentes na mesma GPU.

## 1. GPU: atualizar, instalar RACE e iniciar prepare

```bash
git fetch origin
git switch lean-backends
git pull --ff-only origin lean-backends
python setup_server.py gpu --gpu 3 && python experiment_ops.py start prepare --gpu 3
```

O instalador adiciona as dependências de dados preservando as versões já
instaladas de torch, vLLM, Transformers, tokenizers, Hub, NumPy e safetensors.
Verifica CUDA e as versões mínimas necessárias para os novos modelos.
Não recria o ambiente, não altera `.env` e não reinstala toda a pilha GPU.
Se for um ambiente **novo**, instale a pilha com
`python setup_server.py gpu --gpu 3 --install-gpu-stack` antes de iniciar.

O RACE usa `ehovy/race`, configuração `all`, com os três splits oficiais.
Uma instalação local íntegra é reutilizada, sem baixar novamente.

O prepare executa:

1. recuperação top-1 teste→treino completo, por dataset;
2. respostas das fontes únicas por Phi-2, DeepSeek, Llama, Phi-4-mini, Mistral e Qwen3;
3. julgamento dos fallbacks pelo Llama fixo;
4. autorreflexões simples e complexas de cada estudante.

Uma fonte pode ser pareada com várias questões de teste, mas é respondida e
refletida apenas uma vez por modelo. Nenhuma condição extra foi adicionada.
Split, seis modelos, prompts e limites são congelados na nova run.

O processo roda em segundo plano e continua após desconectar do SSH.
Para verificar:

```bash
python experiment_ops.py status
tail -n 40 .run_state/prepare.log
```

O ID aparece no status assim que o manifesto é criado. Ele fica salvo localmente;
os próximos comandos não exigem copiá-lo manualmente.

## 2. GPU: ao concluir prepare, compartilhar no GitHub

```bash
python experiment_ops.py share prepare
```

O comando verifica a conclusão, empacota pares, respostas, reflexões, manifesto
e **o RACE completo**, faz commit somente desses pacotes e envia para `origin`.
Os arquivos são comprimidos e divididos em partes de até 20 MiB, evitando o
limite por arquivo do GitHub mesmo quando as reflexões brutas são grandes.
Não envia `.env`, credenciais, pesos, cache ou outros arquivos de trabalho.

O pacote fica em `experiment_handoff/<id>/prepare/`; `current.json` identifica
a run compartilhada. Se o push falhar por uma atualização remota, execute
`git pull --rebase origin lean-backends` **apenas se não houver alterações locais
pendentes**, resolva qualquer conflito e repita `git push origin HEAD`.
Não use force push. Um erro de rede no push não exige regenerar o experimento.

## 3. Petrobras: atualizar, instalar o pacote RACE e iniciar teacher

```bash
git fetch origin
git switch lean-backends
git pull --ff-only origin lean-backends
python setup_server.py petrobras && python experiment_ops.py start teacher
```

O instalador restaura e verifica todos os arquivos recebidos da GPU, inclusive
`data/processed/race/{train,validation,test}.jsonl`. Não precisa acessar o
Hugging Face, nem instalar torch, CUDA, vLLM ou datasets nesse servidor.
Instala somente `requirements-azure.txt` e verifica a presença das credenciais
Azure locais. Mantenha endpoint, chave e CA corporativa no `.env` já utilizado.

O teacher lê os itens completos dos pares recebidos. Ele:

1. responde as fontes de treino e gera suas próprias reflexões;
2. gera reflexões simples/complexas sobre as respostas de cada um dos seis estudantes;
3. responde o teste nas suas condições baseline, self_simple e self_complex.

O launcher aplica os limites congelados no manifesto ao processo filho,
sem editar o `.env` da Petrobras. Assim, um default diferente entre servidores
não altera o experimento. A API mantém as credenciais e configurações de rede locais.

```bash
python experiment_ops.py status
tail -n 40 .run_state/teacher.log
```

## 4. Petrobras: ao concluir teacher, devolver pelo GitHub

```bash
python experiment_ops.py share teacher
```

Esse pacote contém as respostas/reflexões do GPT, reflexões externas dos seis
estudantes e o recibo de conclusão. O RACE não é enviado novamente.

## 5. GPU: receber teacher e iniciar finish

```bash
git pull --ff-only origin lean-backends
python experiment_ops.py start finish --gpu 3
```

O launcher restaura o pacote teacher da run compartilhada, verifica a conclusão
da etapa anterior e executa as cinco condições de cada estudante: baseline,
self_simple, self_complex, teacher_simple e teacher_complex. Depois resolve os
fallbacks com o juiz fixo e consolida os resultados com os resultados do GPT.

```bash
python experiment_ops.py status
tail -n 40 .run_state/finish.log
```

## 6. GPU: publicar os resultados finais

```bash
python experiment_ops.py share finish
```

Os resultados originais ficam em
`data/results/reflection_top1/<id>/analysis/all_outcomes.jsonl` e `accuracy.csv`.
Para trazê-los ao computador da análise:

```bash
git pull --ff-only origin lean-backends
python experiment_ops.py restore finish
```

Abra `validation_accuracy_by_similarity.ipynb` e use o ID mostrado por
`python experiment_ops.py status`. O notebook funciona também no teste e inclui
RACE, filtros de falha/contexto/think, middle/high, cobertura e interseção de itens.

## Retomada e identificação explícita

Uma etapa que falhou pode ser retomada repetindo `start` da mesma etapa, depois
de corrigir a causa. Não use `--fresh`: os checkpoints existentes serão mantidos.
Se o worker ainda estiver vivo, o lock impede iniciar outra cópia nesse checkout.

Para trabalhar com uma run específica em vez da indicada em `current.json`:

```bash
python experiment_ops.py start teacher --experiment-id <id>
python experiment_ops.py share teacher --experiment-id <id>
python experiment_ops.py restore finish --experiment-id <id>
```

Antes de atualizar código durante uma etapa, espere ela terminar. Alterar prompts
ou políticas no meio da run é incompatível com o manifesto. Runs antigas mantêm
seus artefatos para análise; o novo prompt do teacher pertence somente à nova run.

`share --pack-only` prepara o pacote sem commit/push. `share` normal faz ambas
as operações. Nenhuma etapa publica automaticamente enquanto ainda está rodando.

## Verificação realizada nesta preparação

- Download e conversão do RACE real: 87.866 treino, 4.887 validação, 4.934 teste.
- Instalação completa do RACE via pacote verificada byte a byte: duas partes,
  29.033.988 bytes comprimidos, revisão `2fec9fd81f1dc971569a9b729c43f2f0e6436637`.
- Com os arquivos locais atuais, após deduplicação: 8.397 questões de teste nos
  cinco datasets. A preparação no servidor registra as contagens definitivas.
- Testes das três etapas com modelos simulados, inclusive fonte top-1 repetida.
- Testes de integridade do transporte, rejeição de pacote corrompido, filtros e contexto.
- Execução em GPU e acesso ao deployment Petrobras devem ser verificados nos
  respectivos servidores; não são simulados como se fossem resultados reais.
