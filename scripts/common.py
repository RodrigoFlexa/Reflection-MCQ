"""
DEPRECADO. O conteúdo mudou para `rmcq/common.py`.

Mantido para compatibilidade com imports antigos. Código novo deve importar de
`rmcq.common`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rmcq.common import *  # noqa: F401,F403,E402
from rmcq.common import (  # noqa: F401,E402
    Extraction,
    Record,
    build_answer_prompt,
    build_eval_prompt,
    build_reflection_prompt,
    build_retrieval_prefix,
    build_retry_prompt,
    cochran_sample_size,
    extract_final_answer,
    format_options,
    format_question,
    iter_jsonl,
    label_distribution,
    make_record,
    read_jsonl,
    stratified_sample,
    strip_think,
    validate_mcq_item,
    write_jsonl,
)
