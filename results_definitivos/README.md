# Results Definitivos

This directory separates the curated validation and test artifacts used for reflection-based MCQ evaluation.

## Validation split
The validation set is stored under `results_definitivos/validation/` and was assembled from the base output at `data/results/reflection_top1/bf7731c4ac1f/self_eval/analysis/all_outcomes.jsonl`.

Key steps:
- Removed all records for `deepseek-r1-0528-qwen3-8b` from the base validation batch.
- Replaced them with `deepseek-r1-distill-llama-8b` rows from `data/results/reflection_top1/91ccab5e5028/analysis/all_outcomes.jsonl`.
- Restricted those replacement rows to `baseline`, `self_simple`, and `self_complex`.
- Preserved `phi2` rows in the final output; if they were absent, they were backfilled from the alternate source using the same validation conditions.
- The resulting `validation/bf7731c4ac1f/analysis/all_outcomes.jsonl` file contains the final validation records and a companion `build_info.json` summarizing the sources and model counts.

## Test split
The `results_definitivos/test/` directory is reserved for a future test split. It is intentionally empty as a placeholder in the validation/test folder structure so the dataset layout remains explicit and reproducible.
