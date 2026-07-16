# journal_extension (nome temporário)

Extensão do paper **"Leveraging LLM Reflection to Improve Small Language Model Agents' Capabilities"** (AGENTICS 2025) para versão de journal. Tudo relacionado ao journal fica isolado nesta pasta — não poluir o resto do repo eagle.

## Estrutura

```
journal_extension/
├── data/mcq/        # datasets MCQ (schema unificado: id, question, choices{label,text}, answerKey)
├── notebooks/       # análises numeradas (01_, 02_, ...)
├── scripts/         # download_datasets.py e futuros scripts do pipeline
├── results/         # saídas de experimentos
└── docs/            # protocolo experimental, decisões
```

## Setup

```bash
pip install pandas pyarrow gdown scikit-learn matplotlib seaborn
python scripts/download_datasets.py          # baixa do Drive (ou --local p/ copiar do sandbox)
python scripts/download_datasets.py --verify # confere integridade
```

Drive dos datasets: https://drive.google.com/drive/folders/1ykTzmLKtYdhmarDvMsdHc1kp7VSdVVWy

## Notebooks

- `01_analise_inicial_datasets.ipynb` — leitura dos datasets, estatísticas de treino/teste, vocabulário, clusters de similaridade das perguntas, análise do impacto esperado da reflexão por dataset e decisões de rota (curto e longo prazo).

Contexto completo do projeto: ver `../CONTEXTO_extensao_journal.md`.
