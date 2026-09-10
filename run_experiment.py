#!/usr/bin/env python
"""Two-server top-1 reflection experiment.

Stages:
  prepare  GPU server: retrieve pairs, answer training sources, self-reflect.
  self-eval GPU server: baseline + self-reflection, no teacher required.
  teacher  Petrobras server: external reflections (optional GPT reference evaluation).
  finish   GPU server: students evaluate with own and teacher reflections.
  status   Any server: report which artifacts are complete.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import multiprocessing
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODELS = (
    "phi2",
    "deepseek-r1-distill-llama-8b",
    "llama3.1-8b",
    "phi4-mini",
    "mistral-7b-instruct",
    "qwen3-8b",
)
DEFAULT_DATASETS = ("aqua", "arc", "logiqa2", "openbookqa", "race")
VALIDATION_MODELS = (
    "phi2", "deepseek-r1-0528-qwen3-8b", "deepseek-r1-distill-qwen-1.5b",
    "llama3.1-8b", "llama3.2-3b", "qwen2.5-3b", "qwen2.5-7b",
    "ministral-3-3b", "ministral-3-8b",
)
# Two GPUs, one process each, no shared engine. The split is by model because a
# vLLM engine is loaded per model anyway: nothing gets loaded that was not
# already loaded once per model, and every generation artifact is already stored
# per model on disk. The experiment id stays derived from the full nine-model
# configuration, so both partitions write into the same run; only the work is
# divided. Balance: p1 carries the heavy 8B reasoning student plus the four
# small ones, p2 carries three 8B instruct students plus the 1.5B reasoning one.
VALIDATION_PARTITIONS: dict[str, tuple[str, ...]] = {
    "p1": ("deepseek-r1-0528-qwen3-8b", "phi2", "llama3.2-3b", "qwen2.5-3b", "ministral-3-3b"),
    "p2": ("deepseek-r1-distill-qwen-1.5b", "llama3.1-8b", "qwen2.5-7b", "ministral-3-8b"),
}
PARTITION_NAMES = tuple(VALIDATION_PARTITIONS)
SELF_CONDITIONS = ("baseline", "self_simple", "self_complex")
EXTERNAL_CONDITIONS = ("teacher_simple", "teacher_complex")
DEFAULT_TEACHER = "gpt-5-4-petrobras"
DEFAULT_JUDGE = "llama3.1-8b"
PIPELINE_VERSION = "top1-two-server-v5"
ANSWER_TEMPERATURE = 0.0
REFLECTION_TEMPERATURE = 0.7
PHI2_TRANSFER_REFLECTION_MAX_TOKENS = 512
# Phi-2 is a base model behind an "Instruct:/Output:" text-completion wrapper,
# not a chat model with a reliable EOS. Once it finishes the real answer it
# tends to keep completing in the style of its pretraining corpus, inventing
# a brand-new exercise instead of stopping. These strings are what that drift
# looks like in practice; cutting generation there keeps the real answer and
# discards nothing usable.
PHI2_STOP_SEQUENCES = ("\nInstruct:", "\nExercise", "\nQuestion:")


def assert_partitions_cover_validation() -> None:
    """A partition that silently drops or duplicates a model would corrupt the run."""
    assigned = [model for models in VALIDATION_PARTITIONS.values() for model in models]
    if sorted(assigned) != sorted(VALIDATION_MODELS):
        raise RuntimeError("VALIDATION_PARTITIONS must partition VALIDATION_MODELS exactly")


def partition_models(part: str | None, models: list[str]) -> list[str]:
    """Models this process executes. `models` stays the full frozen list."""
    if part is None:
        return models
    assert_partitions_cover_validation()
    if sorted(models) != sorted(VALIDATION_MODELS):
        raise ValueError("--part only applies to the frozen nine-model validation grid")
    return [model for model in models if model in VALIDATION_PARTITIONS[part]]


def part_suffix(part: str | None) -> str:
    return "" if part is None else f".{part}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "self-eval", "teacher", "finish", "merge", "status"))
    parser.add_argument("--preset", choices=("validation-threshold",))
    parser.add_argument("--experiment-id", help="Required after prepare; printed by that stage.")
    parser.add_argument("--models")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-role", choices=("reference", "teacher-only"))
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE,
        help=(
            "Fixed model that decides selected-option match when a student's own "
            "answer doesn't parse as FINAL ANSWER: <letter>. Never the student "
            "itself, so a model that ignores formatting instructions doesn't also "
            "grade itself."
        ),
    )
    parser.add_argument("--backend", choices=("vllm", "hf", "stub"), default=None)
    parser.add_argument("--teacher-backend", choices=("azure", "stub"), default="azure")
    parser.add_argument("--gpu", default=os.environ.get("RMCQ_NOTEBOOK_GPU", "0"))
    parser.add_argument("--eval-split", choices=("validation", "test"))
    parser.add_argument("--generation-profile", choices=("legacy", "final"), default="final")
    parser.add_argument("--eval-cap", "--validation-cap", dest="validation_cap", type=int)
    parser.add_argument("--train-cap", type=int, help="Smoke tests only; production must use all training items.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--reflection-temperature", type=float, default=REFLECTION_TEMPERATURE)
    parser.add_argument("--exchange-root", default="experiment_exchange")
    parser.add_argument("--results-root", default="data/results/reflection_top1")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--write-id", type=Path, help="Write the prepared experiment id before generation starts.")
    parser.add_argument("--part", choices=PARTITION_NAMES,
                        help="Run only this partition's models, on this --gpu. The run id is unchanged.")
    for name in PARTITION_NAMES:
        parser.add_argument(f"--{name}", dest="part", action="store_const", const=name,
                            help=f"Shorthand for --part {name}: {', '.join(VALIDATION_PARTITIONS[name])}")
    args = parser.parse_args()
    validation_preset = args.preset == "validation-threshold"
    args.models = args.models or ",".join(VALIDATION_MODELS if validation_preset else DEFAULT_MODELS)
    args.eval_split = args.eval_split or ("validation" if validation_preset else "test")
    args.teacher_role = args.teacher_role or ("teacher-only" if validation_preset else "reference")
    if validation_preset:
        if args.eval_split != "validation" or args.teacher_role != "teacher-only" or args.generation_profile != "final":
            parser.error("validation-threshold requires validation, teacher-only and final generation")
        if args.backend not in (None, "vllm", "stub"):
            parser.error("validation-threshold requires vLLM (stub is only for offline tests)")
        args.backend = args.backend or "vllm"
    if not 0.0 <= args.reflection_temperature <= 2.0:
        parser.error("--reflection-temperature must be between 0.0 and 2.0")
    if args.batch_size <= 0 or any(v is not None and v <= 0 for v in (args.validation_cap, args.train_cap)):
        parser.error("Batch size and optional data caps must be positive")
    if args.part is not None:
        if args.stage in ("teacher", "merge", "status"):
            parser.error(f"--part does not apply to {args.stage}; it splits GPU generation only")
        # Later stages take their model list from the frozen manifest, which is
        # only read in main(); partition_models validates the grid there.
        if validation_preset:
            try:
                partition_models(args.part, split_csv(args.models))
            except (ValueError, RuntimeError) as exc:
                parser.error(str(exc))
    return args


@contextlib.contextmanager
def exclusive(path: Path):
    """Serialize the once-per-run work two partitions would otherwise both do.

    Retrieval, the manifest and the RACE fingerprints are shared by both
    partitions; the per-model generation that follows is not. On POSIX this is a
    real advisory lock on `path`; elsewhere (single-process runs, Windows
    checkouts, tests) it is a no-op, because partitions only ever start from
    experiment_ops, which already refuses to run outside Linux.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        yield
        return
    handle = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        os.close(handle)


def find_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "rmcq").is_dir():
            return candidate
    raise RuntimeError("repository root containing rmcq/ was not found")


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def json_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def normalize_stem(item: dict[str, Any]) -> str:
    value = f"{item.get('context') or ''}\n{item.get('question') or ''}"
    if item.get("dataset") == "race":
        value += "\n" + " | ".join(c["text"] for c in item["choices"])
    return " ".join(value.casefold().split())


def dedupe(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    kept = []
    for item in items:
        stem = normalize_stem(item)
        if stem not in seen:
            seen.add(stem)
            kept.append(item)
    return kept, len(items) - len(kept)


def load_splits(root: Path, datasets: list[str], cap: int | None,
                train_cap: int | None = None, eval_split: str = "validation") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state: dict[str, Any] = {}
    audit = []
    for dataset in datasets:
        folder = root / "data" / "processed" / dataset
        train = load_jsonl(folder / "train.jsonl")
        validation_path = folder / f"{eval_split}.jsonl"
        if not validation_path.exists():
            raise FileNotFoundError(f"Missing {validation_path}; for RACE run python prepare_datasets.py")
        validation = load_jsonl(validation_path)
        for expected_split, items in (("train", train), (eval_split, validation)):
            if any(item["split"] != expected_split for item in items):
                raise ValueError(f"{dataset}: split metadata does not match {expected_split}.jsonl")
            if len({item["uid"] for item in items}) != len(items):
                raise ValueError(f"{dataset}/{expected_split}: duplicate question identifiers")
        if cap is not None and len(validation) > cap:
            validation = random.Random(42).sample(validation, cap)
        train, train_duplicates = dedupe(train)
        validation, validation_duplicates = dedupe(validation)
        validation_stems = {normalize_stem(item) for item in validation}
        before = len(train)
        train = [item for item in train if normalize_stem(item) not in validation_stems]
        cross_split_removed = before - len(train)
        article_overlap_removed = 0
        if dataset == "race":
            eval_articles = {item["article_uid"] for item in validation}
            before_articles = len(train)
            train = [item for item in train if item["article_uid"] not in eval_articles]
            article_overlap_removed = before_articles - len(train)
        if train_cap is not None and len(train) > train_cap:
            train = random.Random(42).sample(train, train_cap)
        if not train or not validation:
            raise ValueError(f"{dataset}: no training candidates or evaluation items after filtering")
        state[dataset] = {"train": train, "validation": validation, "eval_split": eval_split}
        audit.append({
            "dataset": dataset,
            "train": len(train),
            "validation": len(validation),
            "train_duplicates_removed": train_duplicates,
            "validation_duplicates_removed": validation_duplicates,
            "cross_split_removed": cross_split_removed,
            "article_overlap_removed": article_overlap_removed, "eval_split": eval_split,
        })
    return state, audit


def embedding_text(item: dict[str, Any]) -> str:
    options = " | ".join(choice["text"].strip() for choice in item["choices"])
    context = (item.get("context") or "").strip()
    pieces = [piece for piece in (context, item["question"].strip(), options) if piece]
    return "\n".join(pieces)


def retrieve_top1(state: dict[str, Any], model_name: str, device: str) -> list[dict[str, Any]]:
    import gc
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    rows: list[dict[str, Any]] = []
    query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""
    def token_lengths(texts):
        lengths = []
        for start in range(0, len(texts), 128):
            encoded = model.tokenizer(texts[start:start + 128], truncation=False, padding=False)
            lengths.extend(len(ids) for ids in encoded["input_ids"])
        return lengths

    for dataset, splits in state.items():
        train = splits["train"]
        validation = splits["validation"]
        train_texts = [embedding_text(item) for item in train]
        query_texts = [query_prefix + embedding_text(item) for item in validation]
        train_lengths, query_lengths = token_lengths(train_texts), token_lengths(query_texts)
        train_embeddings = model.encode(
            train_texts, batch_size=128,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
        )
        validation_embeddings = model.encode(
            query_texts, batch_size=128,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
        )
        # Chunked multiplication avoids materializing the complete validation x train matrix.
        for start in range(0, len(validation), 256):
            scores = validation_embeddings[start:start + 256] @ train_embeddings.T
            indices = np.argmax(scores, axis=1)
            for offset, source_index in enumerate(indices):
                val_item = validation[start + offset]
                source_item = train[int(source_index)]
                rows.append({
                    "dataset": dataset,
                    "eval_split": splits.get("eval_split", "validation"),
                    "eval_uid": val_item["uid"],
                    "embedding_source_tokens": train_lengths[int(source_index)],
                    "embedding_query_tokens": query_lengths[start + offset],
                    "embedding_max_tokens": model.max_seq_length,
                    "embedding_truncated": max(train_lengths[int(source_index)], query_lengths[start + offset]) > model.max_seq_length,
                    "val_uid": val_item["uid"],
                    "source_uid": source_item["uid"],
                    "similarity": float(scores[offset, source_index]),
                    "validation_item": val_item,
                    "source_item": source_item,
                })
        del train_embeddings, validation_embeddings
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return rows


def cache_key(*parts: str) -> str:
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def training_answer_budget(model_key: str, profile: str = "legacy") -> int:
    if profile == "final":
        from rmcq.generation import final_budget
        return final_budget(model_key, "answer")
    return 512 if model_key == "phi2" else 1024


def training_answer_retry_budget(model_key: str) -> int:
    # DeepSeek-R1-distill spends its budget inside <think>...</think>, which is
    # stripped before saving: a truncated block leaves an empty answer, not a
    # cut-off-but-usable one. Observed p99 among *successful* generations was
    # already 1085/1152 tokens at the old 768/1152 budget, with ~20% still
    # exhausting the retry outright — the tail runs well past 1152, so give it
    # real headroom instead of the default +50% fallback.
    if model_key == "phi2":
        base = training_answer_budget(model_key)
        return base + max(128, base // 2)
    return 2048


def validation_answer_budget(model_key: str, profile: str = "legacy") -> int:
    if profile == "final":
        return training_answer_budget(model_key, profile)
    return 384 if model_key == "phi2" else 512


def judge_budget(model_key: str) -> int:
    # The judge is asked for exactly one word, but it still shares the model's
    # generation quirks: a reasoning model opens <think> here too, unconditionally,
    # even for a one-word classification. The pilot at the old flat 128 budget
    # (never tuned per model) discarded 40% of DeepSeek's judge calls as
    # length_exhausted — same failure shape as the training answer, just never
    # given the same headroom.
    return 128 if model_key == "phi2" else 512


def judge_retry_budget(model_key: str) -> int:
    if model_key == "phi2":
        base = judge_budget(model_key)
        return base + max(128, base // 2)
    return 2048


def reflection_budget(model_key: str, depth: str, profile: str = "legacy") -> int:
    if profile == "final":
        from rmcq.generation import final_budget
        return final_budget(model_key, "reflection")
    if model_key == "phi2":
        return 256 if depth == "simple" else 384
    return 768 if depth == "simple" else 1024


def reflection_retry_budget(model_key: str, depth: str) -> int:
    if model_key == "phi2":
        return 384 if depth == "simple" else 512
    initial = reflection_budget(model_key, depth)
    return initial + initial // 2


def cached_generate(backend: Any, path: Path, prompts: dict[str, str], max_tokens: int,
                    batch_size: int, fresh: bool, description: str,
                    temperature: float = ANSWER_TEMPERATURE,
                    retry_max_tokens: int | None = None,
                    stop: tuple[str, ...] = (), profile: str = "legacy",
                    adapt_to_context: bool = False) -> dict[str, dict[str, Any]]:
    from rmcq.backends.base import GenParams
    if profile == "final":
        from rmcq.generation import generate_once
        return generate_once(backend, path, prompts, max_tokens, batch_size, fresh,
                             description, temperature, stop, adapt_to_context)

    max_len = getattr(backend, "max_len", None)
    if max_len and hasattr(backend, "tokenizer"):
        overflow = []
        for key, prompt in prompts.items():
            prompt_tokens = len(backend.render_token_ids(backend.tokenizer, prompt))
            if prompt_tokens + max_tokens > max_len:
                overflow.append((key, prompt_tokens))
        if overflow:
            key, prompt_tokens = overflow[0]
            raise RuntimeError(
                f"{description}: {len(overflow)} prompt(s) exceed the model context without "
                f"truncation; first key={key!r}, prompt={prompt_tokens}, output={max_tokens}, "
                f"context={max_len}. The retrieved question was not silently removed."
            )

    cached = {} if fresh or not path.exists() else {row["key"]: row for row in load_jsonl(path)}
    missing = []
    for key, prompt in prompts.items():
        if retry_max_tokens is None and not stop:
            # Preserve compatibility with answer/judge checkpoints from v4.
            hash_input = f"{backend.key}\0{max_tokens}\0{temperature}\0{prompt}"
        else:
            hash_input = (
                f"{backend.key}\0{max_tokens}\0{retry_max_tokens}\0{','.join(stop)}\0"
                f"{temperature}\0{prompt}"
            )
        prompt_digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
        current = cached.get(key, {})
        terminal_absence = current.get("finish_reason") in {
            "content_filter", "length_exhausted", "empty_exhausted",
        }
        invalid = (
            not current.get("text") and not terminal_absence
        ) or current.get("finish_reason") == "length"
        if key not in cached or cached[key].get("prompt_hash") != prompt_digest or invalid:
            missing.append((key, prompt, prompt_digest))
    print(f"{description}: total={len(prompts)} cache={len(prompts)-len(missing)} missing={len(missing)}", flush=True)
    checkpoint = min(batch_size, 8) if backend.spec.provider == "ollama" else batch_size
    for start in range(0, len(missing), checkpoint):
        batch = missing[start:start + checkpoint]
        generations = backend.generate(
            [prompt for _, prompt, _ in batch],
            GenParams(max_new_tokens=max_tokens, temperature=temperature, stop=stop),
            desc=f"{description} [{start + 1}-{start + len(batch)}]",
        )
        for (key, _prompt, digest), generation in zip(batch, generations):
            cached[key] = {
                "key": key, "prompt_hash": digest, "text": generation.text,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "finish_reason": generation.finish_reason,
                "max_new_tokens_used": max_tokens,
            }
        save_jsonl(path, cached.values())
        empty_failed = [
            key for key, _prompt, _digest in batch
            if not cached[key]["text"] and cached[key]["finish_reason"] != "content_filter"
        ]
        truncated_failed = [
            key for key, _prompt, _digest in batch
            if cached[key]["finish_reason"] == "length"
        ]
        retryable = set(empty_failed + truncated_failed)
        if retryable and backend.spec.provider != "azure":
            retry_batch = [entry for entry in batch if entry[0] in retryable]
            retry_tokens = retry_max_tokens or (
                max_tokens + max(128, max_tokens // 2)
            )
            if max_len and hasattr(backend, "tokenizer"):
                overflow = []
                for key, prompt, _digest in retry_batch:
                    prompt_tokens = len(backend.render_token_ids(backend.tokenizer, prompt))
                    if prompt_tokens + retry_tokens > max_len:
                        overflow.append((key, prompt_tokens))
                if overflow:
                    overflow_by_key = dict(overflow)
                    overflow_keys = set(overflow_by_key)
                    for key in overflow_keys:
                        cached[key]["text"] = ""
                        cached[key]["finish_reason"] = "length_exhausted"
                        cached[key]["discarded"] = True
                        cached[key]["discard_reason"] = "retry_exceeds_context"
                        cached[key]["retry_prompt_tokens"] = overflow_by_key[key]
                        cached[key]["retry_max_new_tokens"] = retry_tokens
                        cached[key]["model_context_tokens"] = max_len
                    retry_batch = [
                        entry for entry in retry_batch if entry[0] not in overflow_keys
                    ]
                    save_jsonl(path, cached.values())
                    print(
                        f"{description}: discarded {len(overflow_keys)} item(s) because "
                        f"the retry with max_new_tokens={retry_tokens} would exceed "
                        f"context={max_len}",
                        flush=True,
                    )
            if retry_batch:
                print(
                    f"{description}: retrying {len(retry_batch)} truncated/empty "
                    f"generation(s) with max_new_tokens={retry_tokens}", flush=True,
                )
                retried = backend.generate(
                    [prompt for _, prompt, _ in retry_batch],
                    GenParams(max_new_tokens=retry_tokens, temperature=temperature, stop=stop),
                    desc=f"{description} retry",
                )
                for (key, _prompt, digest), generation in zip(retry_batch, retried):
                    cached[key] = {
                        "key": key, "prompt_hash": digest, "text": generation.text,
                        "prompt_tokens": generation.prompt_tokens,
                        "completion_tokens": generation.completion_tokens,
                        "finish_reason": generation.finish_reason,
                        "max_new_tokens_used": retry_tokens,
                    }
                save_jsonl(path, cached.values())
                empty_failed = [
                    key for key, _prompt, _digest in retry_batch if not cached[key]["text"]
                ]
                truncated_failed = [
                    key for key, _prompt, _digest in retry_batch
                    if cached[key]["finish_reason"] == "length"
                ]
            else:
                empty_failed = []
                truncated_failed = []
        if truncated_failed:
            truncated_set = set(truncated_failed)
            for key in truncated_failed:
                cached[key]["text"] = ""
                cached[key]["finish_reason"] = "length_exhausted"
                cached[key]["discarded"] = True
            save_jsonl(path, cached.values())
            empty_failed = [key for key in empty_failed if key not in truncated_set]
            print(
                f"{description}: discarded {len(truncated_failed)} item(s) still "
                "truncated after the final attempt",
                flush=True,
            )
        if empty_failed:
            for key in empty_failed:
                cached[key]["text"] = ""
                cached[key]["finish_reason"] = "empty_exhausted"
                cached[key]["discarded"] = True
            save_jsonl(path, cached.values())
            print(
                f"{description}: discarded {len(empty_failed)} item(s) still empty "
                "after the final attempt",
                flush=True,
            )
    return {key: cached[key] for key in prompts}


def resolve_answers(backend: Any, cache_dir: Path, stage: str, generated: dict[str, dict[str, Any]],
                    items: dict[str, dict[str, Any]], batch_size: int, fresh: bool,
                    profile: str = "legacy") -> dict[str, dict[str, Any]]:
    from rmcq.prompts import build_judge_prompt, extract_final_answer, parse_judge_verdict

    results: dict[str, dict[str, Any]] = {}
    judge_prompts = {}
    for key, row in generated.items():
        if row.get("finish_reason") in {
            "content_filter", "length_exhausted", "empty_exhausted", "prompt_context_exceeded",
        }:
            method = row["finish_reason"]
            results[key] = {
                "selected_answer": None, "correct": None,
                "eval_method": method,
            }
            continue
        answer = extract_final_answer(row["text"])
        if answer is None:
            judge_prompts[key] = build_judge_prompt(items[key], row["text"])
        else:
            results[key] = {"selected_answer": answer, "correct": answer == items[key]["answerKey"], "eval_method": "parser"}
    if judge_prompts:
        judged = cached_generate(
            backend, cache_dir / f"judge_{stage}.jsonl", judge_prompts,
            judge_budget(backend.key), batch_size, fresh, f"judge {stage}",
            retry_max_tokens=None if backend.key == "phi2" else judge_retry_budget(backend.key),
            stop=PHI2_STOP_SEQUENCES if backend.key == "phi2" else (),
            profile=profile,
        )
        for key, row in judged.items():
            if row.get("finish_reason") in {"length_exhausted", "empty_exhausted", "prompt_context_exceeded", "content_filter"}:
                results[key] = {
                    "selected_answer": None, "correct": None,
                    "eval_method": f"judge_{row['finish_reason']}",
                }
                continue
            verdict = parse_judge_verdict(row["text"])
            results[key] = {"selected_answer": None, "correct": verdict,
                            "eval_method": "judge" if verdict is not None else "unresolved"}
    return results


def reflection_status(outputs: dict[str, dict[str, dict[str, Any]]], uid: str) -> dict[str, str]:
    return {
        depth: outputs[depth].get(uid, {}).get("finish_reason", "not_generated")
        for depth in ("simple", "complex")
    }


def unavailable_memory_method(
    attempt_row: dict[str, Any], reflection_row: dict[str, Any] | None, depth: str
) -> str:
    if attempt_row.get("eval_method") in {
        "content_filter", "length_exhausted", "empty_exhausted", "prompt_context_exceeded",
    }:
        return f"source_answer_{attempt_row['eval_method']}"
    if "correct" in attempt_row and attempt_row["correct"] is None:
        return f"source_answer_{attempt_row.get('eval_method') or 'unresolved'}"
    status = (reflection_row or {}).get("reflection_status", {}).get(depth)
    if status in {"content_filter", "length_exhausted", "empty_exhausted", "prompt_context_exceeded"}:
        return f"source_reflection_{status}"
    return "source_reflection_unavailable"


def validation_prompt_issue(
    backend: Any,
    model_key: str,
    prompt: str,
    condition: str,
    reflection: str | None,
    answer_tokens: int,
    profile: str = "legacy",
) -> dict[str, Any] | None:
    """Return why a validation prompt must be skipped, without truncating it."""
    reflection_tokens = None
    if reflection and model_key == "phi2" and profile == "legacy":
        reflection_tokens = backend.count_tokens(reflection)
        if reflection_tokens > PHI2_TRANSFER_REFLECTION_MAX_TOKENS:
            return {
                "eval_method": "reflection_token_limit_exceeded",
                "reflection_tokens": reflection_tokens,
                "reflection_token_limit": PHI2_TRANSFER_REFLECTION_MAX_TOKENS,
            }

    context_tokens = getattr(backend, "max_len", None) or getattr(backend, "num_ctx", None)
    if not context_tokens:
        return None
    if hasattr(backend, "tokenizer"):
        prompt_tokens = len(backend.render_token_ids(backend.tokenizer, prompt))
        token_count_method = "tokenizer"
    else:
        # Ollama owns the tokenizer and chat template, so retain a conservative
        # reserve around the backend's character-based estimate.
        prompt_tokens = backend.count_tokens(prompt) + 128
        token_count_method = "estimate_plus_chat_reserve"
    if prompt_tokens + answer_tokens <= context_tokens:
        return None
    return {
        "eval_method": (
            "validation_context_exceeded"
            if condition == "baseline"
            else "transfer_context_exceeded"
        ),
        "prompt_tokens": prompt_tokens,
        "answer_tokens_reserved": answer_tokens,
        "model_context_tokens": context_tokens,
        "token_count_method": token_count_method,
        **({"reflection_tokens": reflection_tokens} if reflection_tokens is not None else {}),
    }


def unique_sources(pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {pair["source_uid"]: pair["source_item"] for pair in pairs}


def manifest_payload(args: argparse.Namespace) -> dict[str, Any]:
    from rmcq.config import (
        BACKEND, MODELS, MAX_MODEL_LEN, AZURE_MAX_TOKENS, AZURE_REASONING_MIN_TOKENS,
        AZURE_REASONING_EFFORT, SEED, TORCH_DTYPE, VLLM_DETERMINISTIC, VLLM_MAX_NUM_SEQS,
    )
    from rmcq.prompts import (
        ANSWER_PROMPT, STUDENT_REFLECTION_PROMPTS, TEACHER_REFLECTION_PROMPTS, TRANSFER_PROMPT,
    )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "models": split_csv(args.models), "datasets": split_csv(args.datasets),
        "teacher_model": args.teacher_model, "judge_model": args.judge_model,
        "teacher_role": getattr(args, "teacher_role", "reference"),
        "experiment_preset": getattr(args, "preset", None),
        "eval_split": args.eval_split, "generation_profile": args.generation_profile,
        "generation_policy": {
            "length_retries": 0 if args.generation_profile == "final" else 1,
            "keep_incomplete_as_memory": False, "clip_transfer_memory": False,
            "preserve_raw_text": args.generation_profile == "final",
        },
        "model_specs": {
            model: {"repo_id": MODELS[model].repo_id, "extra_kwargs": MODELS[model].extra_kwargs}
            for model in sorted(set(split_csv(args.models) + [args.teacher_model, args.judge_model]))
        },
        "runtime_limits": {"max_model_len": MAX_MODEL_LEN,
                           "generation_seed": SEED, "dtype": TORCH_DTYPE,
                           "vllm_deterministic": VLLM_DETERMINISTIC, "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
                           "azure_max_tokens": AZURE_MAX_TOKENS,
                           "azure_reasoning_min_tokens": AZURE_REASONING_MIN_TOKENS,
                           "azure_reasoning_effort": AZURE_REASONING_EFFORT},
        "validation_cap": args.validation_cap,
        "train_cap": args.train_cap,
        "student_backend": args.backend or BACKEND,
        "generation_temperatures": {
            "student_answer": ANSWER_TEMPERATURE,
            "student_judge": ANSWER_TEMPERATURE,
            "student_reflection": args.reflection_temperature,
            "gpt_5_4_petrobras": "provider_default; temperature omitted",
        },
        "training_answer_max_tokens": {
            model: training_answer_budget(model, args.generation_profile) for model in split_csv(args.models)
        },
        "training_answer_retry_max_tokens": {
            model: training_answer_retry_budget(model) if args.generation_profile == "legacy" else None for model in split_csv(args.models)
        },
        "validation_answer_max_tokens": {
            model: validation_answer_budget(model, args.generation_profile) for model in split_csv(args.models)
        },
        "judge_max_tokens": {
            model: judge_budget(model) for model in split_csv(args.models)
        },
        "judge_retry_max_tokens": {
            model: judge_retry_budget(model) if args.generation_profile == "legacy" else None for model in split_csv(args.models)
        },
        "phi2_stop_sequences": list(PHI2_STOP_SEQUENCES),
        "reflection_max_tokens": {
            model: {
                depth: reflection_budget(model, depth, args.generation_profile) for depth in ("simple", "complex")
            }
            for model in split_csv(args.models)
        },
        "reflection_retry_max_tokens": {
            model: {
                depth: reflection_retry_budget(model, depth) if args.generation_profile == "legacy" else None
                for depth in ("simple", "complex")
            }
            for model in split_csv(args.models)
        },
        "embedding_model": args.embedding_model,
        "answer_prompt": ANSWER_PROMPT,
        "student_reflection_prompts": STUDENT_REFLECTION_PROMPTS,
        "teacher_reflection_prompts": TEACHER_REFLECTION_PROMPTS,
        "transfer_prompt": TRANSFER_PROMPT,
        "thinking_policy": "request disabled when supported; strip embedded <think> blocks",
        "seed": 42,
    }


def assert_manifest_compatible(manifest: dict[str, Any]) -> None:
    """Refuse to mix artifacts made with different code/prompt revisions."""
    from rmcq.prompts import (
        ANSWER_PROMPT, STUDENT_REFLECTION_PROMPTS, TEACHER_REFLECTION_PROMPTS, TRANSFER_PROMPT,
    )
    expected = {
        "pipeline_version": PIPELINE_VERSION,
        "answer_prompt": ANSWER_PROMPT,
        "student_reflection_prompts": STUDENT_REFLECTION_PROMPTS,
        "teacher_reflection_prompts": TEACHER_REFLECTION_PROMPTS,
        "transfer_prompt": TRANSFER_PROMPT,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "The exchange was created with incompatible code or prompts: "
            + ", ".join(mismatches)
        )


def find_compatible_pair_exchange(
    exchange: Path, datasets: list[str], args: argparse.Namespace
) -> Path | None:
    """Find an older run whose retrieval inputs are exactly compatible."""
    expected = {
        "datasets": datasets,
        "validation_cap": args.validation_cap,
        "train_cap": args.train_cap,
        "embedding_model": args.embedding_model,
        "seed": 42,
    }
    if hasattr(args, "eval_split"):
        expected["eval_split"] = args.eval_split
    if hasattr(args, "data_fingerprints"):
        expected["data_fingerprints"] = args.data_fingerprints
    if not exchange.parent.exists():
        return None
    manifests = sorted(
        exchange.parent.glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest_path in manifests:
        candidate = manifest_path.parent.resolve()
        if candidate == exchange.resolve():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if any(manifest.get(key) != value for key, value in expected.items()):
            continue
        if all((candidate / "pairs" / f"{dataset}.jsonl").exists() for dataset in datasets):
            return candidate
    return None


def stage_prepare(root: Path, exchange: Path, results: Path, args: argparse.Namespace) -> None:
    from rmcq.backends import get_backend
    from rmcq.prompts import REFLECTION_DEPTHS, build_answer_prompt, build_reflection_prompt

    datasets = split_csv(args.datasets)
    part = getattr(args, "part", None)
    compatible: Path | None = None
    pair_paths = [exchange / "pairs" / f"{dataset}.jsonl" for dataset in datasets]
    # Retrieval is identical for every partition and must happen exactly once.
    # Whichever partition arrives first computes it; the other waits here and
    # then reads the same frozen pairs, instead of embedding the corpus twice.
    with exclusive(exchange / "pairs.lock"):
        if not args.fresh and all(path.exists() for path in pair_paths):
            pairs = load_pairs(exchange, datasets)
            print(f"top-1 retrieval: reused {len(pairs)} cached pairs", flush=True)
        else:
            compatible = None if args.fresh else find_compatible_pair_exchange(exchange, datasets, args)
            if compatible is not None:
                for dataset in datasets:
                    destination = exchange / "pairs" / f"{dataset}.jsonl"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(compatible / "pairs" / f"{dataset}.jsonl", destination)
                audit_source = compatible / "retrieval_audit.json"
                if audit_source.exists():
                    shutil.copy2(audit_source, exchange / "retrieval_audit.json")
                pairs = load_pairs(exchange, datasets)
                print(
                    f"top-1 retrieval: reused {len(pairs)} compatible pairs from "
                    f"{compatible.name}",
                    flush=True,
                )
            else:
                state, audit = load_splits(root, datasets, args.validation_cap, args.train_cap, args.eval_split)
                pairs = retrieve_top1(state, args.embedding_model, args.embedding_device)
                for dataset in datasets:
                    save_jsonl(
                        exchange / "pairs" / f"{dataset}.jsonl",
                        [p for p in pairs if p["dataset"] == dataset],
                    )
                save_json(exchange / "retrieval_audit.json", audit)

    sources = unique_sources(pairs)
    content_filter_count = 0
    models = partition_models(part, split_csv(args.models))
    if part is not None:
        print(f"partition {part}: {len(models)} of {len(split_csv(args.models))} students "
              f"on GPU {args.gpu}: {', '.join(models)}", flush=True)
    model_caches: dict[str, Path] = {}
    generated_by_model: dict[str, dict[str, dict[str, Any]]] = {}

    # Pass 1: each model generates its own training answers. Nobody is judged yet.
    for model_key in models:
        model_cache = results / "work" / "prepare" / model_key
        model_caches[model_key] = model_cache
        if compatible is not None and not args.fresh:
            previous_cache = results.parent / compatible.name / "work" / "prepare" / model_key
            reused = []
            for filename in ("train_answers.jsonl", "judge_train.jsonl"):
                source = previous_cache / filename
                destination = model_cache / filename
                if source.exists() and not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    reused.append(filename)
            if reused:
                print(
                    f"{model_key}: reused compatible prepare checkpoints from "
                    f"{compatible.name}: {', '.join(reused)}",
                    flush=True,
                )
        answer_prompts = {uid: build_answer_prompt(item) for uid, item in sources.items()}
        with get_backend(model_key, kind=args.backend) as backend:
            generated_by_model[model_key] = cached_generate(
                backend, model_cache / "train_answers.jsonl", answer_prompts,
                training_answer_budget(model_key, args.generation_profile), args.batch_size, args.fresh,
                f"{model_key} training answers",
                retry_max_tokens=None if model_key == "phi2" else training_answer_retry_budget(model_key),
                stop=PHI2_STOP_SEQUENCES if model_key == "phi2" else (),
                profile=args.generation_profile,
            )

    # Pass 2: one fixed judge model decides selected-option match for every
    # model's unparsed answers, loaded once. A student never grades itself —
    # see docs/experiment_protocol.md for why (a self-graded DeepSeek barely
    # follows "respond in exactly one word" either).
    verdicts_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    with get_backend(args.judge_model, kind=args.backend) as judge_backend:
        for model_key in models:
            verdicts_by_model[model_key] = resolve_answers(
                judge_backend, model_caches[model_key], "train",
                generated_by_model[model_key], sources, args.batch_size, args.fresh,
                profile=args.generation_profile,
            )

    # Pass 3: each model reflects on its own (now-judged) training answers.
    for model_key in models:
        model_cache = model_caches[model_key]
        generated = generated_by_model[model_key]
        verdicts = verdicts_by_model[model_key]
        with get_backend(model_key, kind=args.backend) as backend:
            reflection_outputs: dict[str, dict[str, dict[str, Any]]] = {}
            for depth in REFLECTION_DEPTHS:
                prompts = {
                    uid: build_reflection_prompt(sources[uid], generated[uid]["text"], verdicts[uid]["correct"], depth, "student")
                    for uid in sources if verdicts[uid]["correct"] is not None
                }
                reflection_outputs[depth] = cached_generate(
                    backend, model_cache / f"self_{depth}.jsonl", prompts,
                    reflection_budget(model_key, depth, args.generation_profile),
                    args.batch_size, args.fresh, f"{model_key} self reflection {depth}",
                    temperature=args.reflection_temperature,
                    retry_max_tokens=reflection_retry_budget(model_key, depth),
                    stop=PHI2_STOP_SEQUENCES if model_key == "phi2" else (),
                    profile=args.generation_profile, adapt_to_context=model_key == "phi2",
                )
        rows = []
        for uid, item in sources.items():
            verdict = verdicts[uid]
            rows.append({
                "dataset": item["dataset"], "source_uid": uid, "item": item,
                "response": generated[uid]["text"],
                "answer_finish_reason": generated[uid]["finish_reason"], **verdict,
                "reflections": {
                    depth: reflection_outputs[depth].get(uid, {}).get("text")
                    for depth in REFLECTION_DEPTHS
                },
                "reflection_status": reflection_status(reflection_outputs, uid),
                "answer_generation": generated[uid],
                "reflection_generations": {depth: reflection_outputs[depth].get(uid, {}) for depth in REFLECTION_DEPTHS},
            })
        content_filter_count += sum(
            row["answer_finish_reason"] == "content_filter"
            for row in rows
        ) + sum(
            status == "content_filter"
            for row in rows for status in row["reflection_status"].values()
        )
        save_jsonl(exchange / "students" / model_key / "train.jsonl", rows)
    # A partition receipt only certifies its own students. `merge` turns the set
    # of partition receipts into the run-wide prepare_receipt.json.
    save_json(exchange / f"prepare_receipt{part_suffix(part)}.json", {
        "pairs": len(pairs), "unique_training_sources": len(sources),
        "part": part, "student_models": models, "judge_model": args.judge_model,
        "content_filter_events": content_filter_count, "complete": True,
    })


def load_pairs(exchange: Path, datasets: list[str]) -> list[dict[str, Any]]:
    return [row for dataset in datasets for row in load_jsonl(exchange / "pairs" / f"{dataset}.jsonl")]


def stage_teacher_only(exchange: Path, results: Path, args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    """GPT only reflects on students' training attempts; no GPT answers or judge calls."""
    from rmcq.backends import get_backend
    from rmcq.prompts import REFLECTION_DEPTHS, build_reflection_prompt

    models, teacher_model = manifest["models"], manifest["teacher_model"]
    sources = unique_sources(load_pairs(exchange, manifest["datasets"]))
    cache_dir = results / "work" / "teacher"
    with get_backend(teacher_model, kind=args.teacher_backend) as backend:
        teacher_rows_by_model: dict[str, list[dict[str, Any]]] = {}
        for student_model in models:
            student_rows = load_jsonl(exchange / "students" / student_model / "train.jsonl")
            student_by_uid = {row["source_uid"]: row for row in student_rows}
            outputs: dict[str, dict[str, dict[str, Any]]] = {}
            for depth in REFLECTION_DEPTHS:
                teacher_prompts = {
                    uid: build_reflection_prompt(sources[uid], row["response"], row["correct"], depth, "teacher")
                    for uid, row in student_by_uid.items() if row["correct"] is not None
                }
                outputs[depth] = cached_generate(
                    backend, cache_dir / student_model / f"teacher_{depth}.jsonl", teacher_prompts,
                    reflection_budget(teacher_model, depth, "final") if args.generation_profile == "final" else (1024 if depth == "simple" else 2048), args.batch_size, args.fresh,
                    f"teacher reflection for {student_model} {depth}",
                    temperature=args.reflection_temperature,
                    profile=args.generation_profile,
                )
            teacher_rows_by_model[student_model] = [{
                "dataset": sources[uid]["dataset"], "source_uid": uid,
                "student_model": student_model,
                "reflections": {depth: outputs[depth].get(uid, {}).get("text") for depth in REFLECTION_DEPTHS},
                "reflection_status": reflection_status(outputs, uid),
                "reflection_generations": {depth: outputs[depth].get(uid, {}) for depth in REFLECTION_DEPTHS},
            } for uid in student_by_uid]

    for model, rows in teacher_rows_by_model.items():
        save_jsonl(exchange / "teacher" / "student_reflections" / f"{model}.jsonl", rows)
    save_json(exchange / "teacher_receipt.json", {
        "teacher_model": teacher_model, "teacher_role": "teacher-only",
        "training_sources": len(sources), "validation_generations": 0,
        "training_answer_generations": 0, "student_models_taught": models,
        "content_filter_events": sum(status == "content_filter" for rows in teacher_rows_by_model.values()
            for row in rows for status in row["reflection_status"].values()),
        "complete": True,
    })


def stage_teacher(exchange: Path, results: Path, args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    if manifest.get("teacher_role", "reference") == "teacher-only":
        return stage_teacher_only(exchange, results, args, manifest)
    from rmcq.backends import get_backend
    from rmcq.prompts import (
        REFLECTION_DEPTHS, build_answer_prompt, build_reflection_prompt, build_transfer_prompt,
    )

    datasets, models = manifest["datasets"], manifest["models"]
    teacher_model = manifest["teacher_model"]
    pairs = load_pairs(exchange, datasets)
    sources = unique_sources(pairs)
    validation = {pair["val_uid"]: pair["validation_item"] for pair in pairs}
    pair_by_val = {pair["val_uid"]: pair for pair in pairs}
    cache_dir = results / "work" / "teacher"
    with get_backend(teacher_model, kind=args.teacher_backend) as backend:
        prompts = {uid: build_answer_prompt(item) for uid, item in sources.items()}
        generated = cached_generate(backend, cache_dir / "train_answers.jsonl", prompts,
                                    training_answer_budget(teacher_model, args.generation_profile),
                                    args.batch_size, args.fresh, "teacher training answers", profile=args.generation_profile)
        verdicts = resolve_answers(backend, cache_dir, "train", generated, sources,
                                   args.batch_size, args.fresh, profile=args.generation_profile)
        self_reflections: dict[str, dict[str, dict[str, Any]]] = {}
        for depth in REFLECTION_DEPTHS:
            reflection_prompts = {
                uid: build_reflection_prompt(sources[uid], generated[uid]["text"], verdicts[uid]["correct"], depth, "student")
                for uid in sources if verdicts[uid]["correct"] is not None
            }
            self_reflections[depth] = cached_generate(
                backend, cache_dir / f"self_{depth}.jsonl", reflection_prompts,
                reflection_budget(teacher_model, depth, "final") if args.generation_profile == "final" else (1024 if depth == "simple" else 2048), args.batch_size, args.fresh,
                f"teacher self reflection {depth}",
                temperature=args.reflection_temperature,
                profile=args.generation_profile,
            )

        teacher_rows_by_model: dict[str, list[dict[str, Any]]] = {}
        for student_model in models:
            student_rows = load_jsonl(exchange / "students" / student_model / "train.jsonl")
            student_by_uid = {row["source_uid"]: row for row in student_rows}
            outputs: dict[str, dict[str, dict[str, Any]]] = {}
            for depth in REFLECTION_DEPTHS:
                teacher_prompts = {
                    uid: build_reflection_prompt(sources[uid], row["response"], row["correct"], depth, "teacher")
                    for uid, row in student_by_uid.items() if row["correct"] is not None
                }
                outputs[depth] = cached_generate(
                    backend, cache_dir / student_model / f"teacher_{depth}.jsonl", teacher_prompts,
                    reflection_budget(teacher_model, depth, "final") if args.generation_profile == "final" else (1024 if depth == "simple" else 2048), args.batch_size, args.fresh,
                    f"teacher reflection for {student_model} {depth}",
                    temperature=args.reflection_temperature,
                    profile=args.generation_profile,
                )
            teacher_rows_by_model[student_model] = [{
                "dataset": sources[uid]["dataset"], "source_uid": uid,
                "student_model": student_model,
                "reflections": {depth: outputs[depth].get(uid, {}).get("text") for depth in REFLECTION_DEPTHS},
                "reflection_status": reflection_status(outputs, uid),
                "reflection_generations": {depth: outputs[depth].get(uid, {}) for depth in REFLECTION_DEPTHS},
            } for uid in student_by_uid]

        condition_prompts: dict[str, str] = {}
        condition_items: dict[str, dict[str, Any]] = {}
        condition_meta: dict[str, dict[str, Any]] = {}
        unavailable_validation_rows: list[dict[str, Any]] = []
        for uid, item in validation.items():
            pair = pair_by_val[uid]
            source_uid = pair["source_uid"]
            for condition in ("baseline", "self_simple", "self_complex"):
                key = cache_key(item["dataset"], uid, condition)
                if condition == "baseline":
                    prompt = build_answer_prompt(item)
                else:
                    depth = condition.removeprefix("self_")
                    reflection = self_reflections[depth].get(source_uid, {}).get("text")
                    if not reflection:
                        source_attempt = verdicts[source_uid] | {
                            "reflection_status": reflection_status(self_reflections, source_uid)
                        }
                        unavailable_validation_rows.append({
                            "model": teacher_model, "dataset": pair["dataset"],
                            "val_uid": pair["val_uid"], "source_uid": source_uid,
                            "similarity": pair["similarity"], "condition": condition,
                            "response": "", "finish_reason": "not_generated",
                            "selected_answer": None, "correct": None,
                            "eval_method": unavailable_memory_method(
                                source_attempt, source_attempt, depth
                            ),
                        })
                        continue
                    prompt = build_transfer_prompt(
                        item, pair["source_item"], generated[source_uid]["text"],
                        verdicts[source_uid]["correct"], reflection,
                    )
                condition_prompts[key] = prompt
                condition_items[key] = item
                condition_meta[key] = {"condition": condition, "pair": pair}
        condition_generated = cached_generate(
            backend, cache_dir / f"{args.eval_split}.jsonl", condition_prompts,
            training_answer_budget(teacher_model, args.generation_profile),
            args.batch_size, args.fresh, f"teacher {args.eval_split}",
            profile=args.generation_profile,
        )
        condition_verdicts = resolve_answers(
            backend, cache_dir, args.eval_split, condition_generated, condition_items,
            args.batch_size, args.fresh,
            profile=args.generation_profile,
        )

    teacher_train = [{
        "dataset": item["dataset"], "source_uid": uid, "item": item,
        "response": generated[uid]["text"],
        "answer_finish_reason": generated[uid]["finish_reason"], **verdicts[uid],
        "reflections": {depth: self_reflections[depth].get(uid, {}).get("text") for depth in REFLECTION_DEPTHS},
        "reflection_status": reflection_status(self_reflections, uid),
        "answer_generation": generated[uid],
        "reflection_generations": {depth: self_reflections[depth].get(uid, {}) for depth in REFLECTION_DEPTHS},
    } for uid, item in sources.items()]
    save_jsonl(exchange / "teacher" / "train.jsonl", teacher_train)
    for model, rows in teacher_rows_by_model.items():
        save_jsonl(exchange / "teacher" / "student_reflections" / f"{model}.jsonl", rows)
    validation_rows = list(unavailable_validation_rows)
    for key, meta in condition_meta.items():
        pair = meta["pair"]
        validation_rows.append({
            "model": teacher_model, "dataset": pair["dataset"], "val_uid": pair["val_uid"],
            "source_uid": pair["source_uid"], "similarity": pair["similarity"],
            "condition": meta["condition"], "response": condition_generated[key]["text"],
            "finish_reason": condition_generated[key]["finish_reason"],
            "evaluation_generation": condition_generated[key],
            **condition_verdicts[key],
        })
    from rmcq.analysis import annotate_outcomes
    validation_rows = annotate_outcomes(validation_rows, pairs,
        {teacher_model: {row["source_uid"]: row for row in teacher_train}}, {})
    save_jsonl(exchange / "teacher" / f"{args.eval_split}.jsonl", validation_rows)
    teacher_filter_events = (
        sum(row["answer_finish_reason"] == "content_filter" for row in teacher_train)
        + sum(status == "content_filter" for row in teacher_train for status in row["reflection_status"].values())
        + sum(
            status == "content_filter"
            for rows in teacher_rows_by_model.values()
            for row in rows for status in row["reflection_status"].values()
        )
        + sum(row.get("finish_reason") == "content_filter" for row in validation_rows)
    )
    save_json(exchange / "teacher_receipt.json", {
        "teacher_model": teacher_model, "training_sources": len(sources),
        "validation_generations": len(validation_rows), "student_models_taught": models,
        "content_filter_events": teacher_filter_events,
        "unavailable_validation_conditions": sum(
            row.get("correct") is None for row in validation_rows
        ),
        "complete": True,
    })


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["dataset"], row["condition"]), []).append(row)
        groups.setdefault((row["model"], "ALL", row["condition"]), []).append(row)
    summary = []
    for (model, dataset, condition), group in sorted(groups.items()):
        resolved = [row for row in group if row.get("correct") is not None]
        summary.append({
            "model": model, "dataset": dataset, "condition": condition,
            "n": len(group), "resolved": len(resolved),
            "coverage": len(resolved) / len(group),
            "accuracy": (sum(bool(row["correct"]) for row in resolved) / len(resolved)) if resolved else None,
            "accuracy_all": sum(bool(row["correct"]) for row in resolved) / len(group),
        })
    return summary


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage_finish(exchange: Path, results: Path, args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    from rmcq.backends import get_backend
    from rmcq.prompts import build_answer_prompt, build_transfer_prompt

    self_only = getattr(args, "stage", "finish") == "self-eval"
    part = getattr(args, "part", None)
    self_root = results / "self_eval"
    destination = self_root if self_only else results
    # With partitions the run-wide receipt only exists after `merge`, so each
    # partition reads back its own snapshot instead.
    self_receipt = self_root / f"self_eval_receipt{part_suffix(part)}.json"
    reuse_self = (not self_only and not args.fresh and self_receipt.exists()
                  and json.loads(self_receipt.read_text(encoding="utf-8")).get("complete") is True)
    pairs = load_pairs(exchange, manifest["datasets"])
    models = partition_models(part, manifest["models"])
    if part is not None:
        print(f"partition {part}: {len(models)} of {len(manifest['models'])} students "
              f"on GPU {args.gpu}: {', '.join(models)}", flush=True)
    student_rows_by_model = {
        model: {row["source_uid"]: row for row in load_jsonl(exchange / "students" / model / "train.jsonl")}
        for model in models
    }
    teacher_rows_by_model = {} if self_only else {
        model: {row["source_uid"]: row for row in load_jsonl(exchange / "teacher" / "student_reflections" / f"{model}.jsonl")}
        for model in models
    }
    # The GPT reference rows belong to the run, not to a partition: with
    # partitions they are attached once, by merge.
    all_rows = (load_jsonl(exchange / "teacher" / f"{args.eval_split}.jsonl")
                if not self_only and part is None
                and manifest.get("teacher_role", "reference") == "reference" else [])
    cache_dirs: dict[str, Path] = {}
    generated_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    items_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    metadata_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    unavailable_rows_by_model: dict[str, list[dict[str, Any]]] = {}
    preserved_rows = {}

    # Pass 1: each model answers its own validation prompts. Nobody is judged yet.
    for model in models:
        preserved_rows[model] = (load_jsonl(self_root / "models" / model / f"{args.eval_split}.jsonl")
                                 if reuse_self else [])
        if reuse_self:
            expected_keys = {(p["val_uid"], c) for p in pairs for c in SELF_CONDITIONS}
            actual_keys = [(r["val_uid"], r["condition"]) for r in preserved_rows[model]]
            if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
                raise RuntimeError(f"Incomplete self-eval snapshot for {model}; cannot complete this run")
        conditions = SELF_CONDITIONS if self_only else (EXTERNAL_CONDITIONS if reuse_self else SELF_CONDITIONS + EXTERNAL_CONDITIONS)
        prompts: dict[str, str] = {}
        items: dict[str, dict[str, Any]] = {}
        metadata: dict[str, dict[str, Any]] = {}
        unavailable_rows: list[dict[str, Any]] = []
        for pair in pairs:
            val_item, source_item = pair["validation_item"], pair["source_item"]
            source_uid = pair["source_uid"]
            for condition in conditions:
                key = cache_key(pair["dataset"], pair["val_uid"], condition)
                reflection = None
                if condition == "baseline":
                    prompt = build_answer_prompt(val_item)
                else:
                    author, depth = condition.split("_", 1)
                    attempt_row = student_rows_by_model[model].get(source_uid)
                    reflection_row = attempt_row if author == "self" else teacher_rows_by_model[model].get(source_uid)
                    reflection = (reflection_row or {}).get("reflections", {}).get(depth)
                    if not reflection:
                        method = (
                            "source_attempt_unavailable" if attempt_row is None
                            else unavailable_memory_method(attempt_row, reflection_row, depth)
                        )
                        unavailable_rows.append({
                            "model": model, "dataset": pair["dataset"],
                            "val_uid": pair["val_uid"], "source_uid": source_uid,
                            "similarity": pair["similarity"], "condition": condition,
                            "response": "", "finish_reason": "not_generated",
                            "selected_answer": None, "correct": None,
                            "eval_method": method,
                        })
                        continue
                    prompt = build_transfer_prompt(
                        val_item, source_item, attempt_row["response"],
                        attempt_row["correct"], reflection,
                    )
                prompts[key], items[key] = prompt, val_item
                metadata[key] = {
                    "condition": condition, "pair": pair, "reflection": reflection,
                }
        cache_dir = results / "work" / "finish" / model
        cache_dirs[model] = cache_dir
        with get_backend(model, kind=args.backend) as backend:
            answer_tokens = validation_answer_budget(model, args.generation_profile)
            accepted_prompts: dict[str, str] = {}
            accepted_items: dict[str, dict[str, Any]] = {}
            accepted_metadata: dict[str, dict[str, Any]] = {}
            for key, prompt in prompts.items():
                meta = metadata[key]
                issue = validation_prompt_issue(
                    backend, model, prompt, meta["condition"], meta["reflection"],
                    answer_tokens,
                    profile=args.generation_profile,
                )
                if issue is None:
                    accepted_prompts[key] = prompt
                    accepted_items[key] = items[key]
                    accepted_metadata[key] = meta
                    continue
                pair = meta["pair"]
                unavailable_rows.append({
                    "model": model, "dataset": pair["dataset"],
                    "val_uid": pair["val_uid"], "source_uid": pair["source_uid"],
                    "similarity": pair["similarity"], "condition": meta["condition"],
                    "response": "", "finish_reason": "not_generated",
                    "selected_answer": None, "correct": None, **issue,
                })
            prompts, items, metadata = accepted_prompts, accepted_items, accepted_metadata
            generated_by_model[model] = cached_generate(
                backend, cache_dir / f"{args.eval_split}.jsonl", prompts, answer_tokens,
                args.batch_size, args.fresh, f"{model} {args.eval_split}",
                stop=PHI2_STOP_SEQUENCES if model == "phi2" else (),
                profile=args.generation_profile,
            )
        items_by_model[model] = items
        metadata_by_model[model] = metadata
        unavailable_rows_by_model[model] = unavailable_rows

    # Pass 2: the same fixed judge model used in `prepare` resolves every
    # model's unparsed validation answers, loaded once.
    verdicts_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    with get_backend(args.judge_model, kind=args.backend) as judge_backend:
        for model in models:
            verdicts_by_model[model] = resolve_answers(
                judge_backend, cache_dirs[model], args.eval_split,
                generated_by_model[model], items_by_model[model],
                args.batch_size, args.fresh,
                profile=args.generation_profile,
            )

    for model in models:
        generated = generated_by_model[model]
        metadata = metadata_by_model[model]
        verdicts = verdicts_by_model[model]
        model_rows = list(unavailable_rows_by_model[model])
        for key, meta in metadata.items():
            pair = meta["pair"]
            model_rows.append({
                "model": model, "dataset": pair["dataset"], "val_uid": pair["val_uid"],
                "source_uid": pair["source_uid"], "similarity": pair["similarity"],
                "condition": meta["condition"], "response": generated[key]["text"],
                "finish_reason": generated[key]["finish_reason"],
                "evaluation_generation": generated[key],
                **verdicts[key],
            })
        from rmcq.analysis import annotate_outcomes
        model_rows = annotate_outcomes(model_rows, pairs, student_rows_by_model, teacher_rows_by_model)
        model_rows = preserved_rows[model] + model_rows
        save_jsonl(destination / "models" / model / f"{args.eval_split}.jsonl", model_rows)
        all_rows.extend(model_rows)
    suffix = part_suffix(part)
    # A partition writes only its own slice of the analysis; the canonical
    # all_outcomes.jsonl and accuracy.csv are written by merge, once both
    # partitions are on disk, so a half-finished run never looks whole.
    save_jsonl(destination / "analysis" / f"all_outcomes{suffix}.jsonl", all_rows)
    summary = summarize(all_rows)
    save_csv(destination / "analysis" / f"accuracy{suffix}.csv", summary)
    filter_audit = [
        row for row in all_rows if "content_filter" in (row.get("eval_method") or "")
    ]
    save_jsonl(destination / "analysis" / f"content_filter_audit{suffix}.jsonl", filter_audit)
    name = "self_eval_receipt" if self_only else "finish_receipt"
    save_json(destination / f"{name}{suffix}.json", {
        "experiment_id": manifest.get("experiment_id"), "eval_split": args.eval_split,
        "conditions": list(SELF_CONDITIONS if self_only else SELF_CONDITIONS + EXTERNAL_CONDITIONS),
        "self_snapshot_reused": bool(reuse_self), "part": part, "models": models,
        "rows": len(all_rows), "content_filter_affected_conditions": len(filter_audit),
        "unresolved_conditions": sum(row.get("correct") is None for row in all_rows),
        "complete": True,
    })
    print(f"completed: {destination}", flush=True)
    if part is not None:
        print(f"partition {part} done; run merge once the other partition finishes.", flush=True)


def merge_receipts(exchange: Path, results: Path, phase: str) -> list[dict[str, Any]]:
    location = {
        "prepare": (exchange, "prepare_receipt"),
        "self-eval": (results / "self_eval", "self_eval_receipt"),
        "finish": (results, "finish_receipt"),
    }.get(phase)
    if location is None:  # the teacher stage is never partitioned
        return []
    folder, name = location
    return [json.loads(path.read_text(encoding="utf-8"))
            for part in PARTITION_NAMES
            for path in [folder / f"{name}.{part}.json"] if path.exists()]


def merge_phase(exchange: Path, results: Path, args: argparse.Namespace,
                manifest: dict[str, Any], phase: str) -> dict[str, Any]:
    """Turn per-partition artifacts into the run-wide ones, or say what is missing.

    Merging is a pure file operation: nothing is regenerated, and a phase is
    only declared complete when every model in the frozen manifest is present.
    """
    split = manifest["eval_split"]
    models = manifest["models"]
    receipts = merge_receipts(exchange, results, phase)
    if not receipts:
        return {"phase": phase, "status": "no_partition_receipts"}
    if phase == "prepare":
        missing = [m for m in models if not (exchange / "students" / m / "train.jsonl").exists()]
        if missing:
            return {"phase": phase, "status": "incomplete", "missing_models": missing}
        save_json(exchange / "prepare_receipt.json", {
            "pairs": receipts[0]["pairs"],
            "unique_training_sources": receipts[0]["unique_training_sources"],
            "student_models": models, "judge_model": manifest["judge_model"],
            "content_filter_events": sum(r["content_filter_events"] for r in receipts),
            "merged_from": [r["part"] for r in receipts], "complete": True,
        })
        return {"phase": phase, "status": "merged", "models": len(models)}

    destination = results / "self_eval" if phase == "self-eval" else results
    missing = [m for m in models if not (destination / "models" / m / f"{split}.jsonl").exists()]
    if missing:
        return {"phase": phase, "status": "incomplete", "missing_models": missing}
    reference = (load_jsonl(exchange / "teacher" / f"{split}.jsonl")
                 if phase == "finish" and manifest.get("teacher_role", "reference") == "reference"
                 and (exchange / "teacher" / f"{split}.jsonl").exists() else [])
    all_rows = list(reference)
    for model in models:
        all_rows.extend(load_jsonl(destination / "models" / model / f"{split}.jsonl"))
    save_jsonl(destination / "analysis" / "all_outcomes.jsonl", all_rows)
    save_csv(destination / "analysis" / "accuracy.csv", summarize(all_rows))
    filter_audit = [row for row in all_rows if "content_filter" in (row.get("eval_method") or "")]
    save_jsonl(destination / "analysis" / "content_filter_audit.jsonl", filter_audit)
    self_only = phase == "self-eval"
    save_json(destination / ("self_eval_receipt.json" if self_only else "finish_receipt.json"), {
        "experiment_id": manifest.get("experiment_id"), "eval_split": split,
        "conditions": list(SELF_CONDITIONS if self_only else SELF_CONDITIONS + EXTERNAL_CONDITIONS),
        "self_snapshot_reused": all(r.get("self_snapshot_reused") for r in receipts),
        "rows": len(all_rows), "content_filter_affected_conditions": len(filter_audit),
        "unresolved_conditions": sum(row.get("correct") is None for row in all_rows),
        "merged_from": [r["part"] for r in receipts], "complete": True,
    })
    return {"phase": phase, "status": "merged", "models": len(models), "rows": len(all_rows)}


def stage_merge(exchange: Path, results: Path, args: argparse.Namespace,
                manifest: dict[str, Any]) -> None:
    report = [merge_phase(exchange, results, args, manifest, phase)
              for phase in ("prepare", "self-eval", "finish")]
    for entry in report:
        detail = ""
        if entry["status"] == "incomplete":
            detail = f" (waiting for {', '.join(entry['missing_models'])})"
        elif entry["status"] == "merged":
            detail = f" ({entry['models']} models" + (f", {entry['rows']} rows)" if "rows" in entry else ")")
        print(f"{entry['phase']}: {entry['status']}{detail}", flush=True)
    save_json(results / "merge_report.json", {"experiment_id": manifest.get("experiment_id"),
                                              "phases": report})
    if all(entry["status"] != "merged" for entry in report):
        raise RuntimeError("Nothing to merge yet; run the partitions first, or check their logs.")


def stage_status(exchange: Path, results: Path, manifest: dict[str, Any] | None) -> None:
    print(f"exchange: {exchange}")
    print(f"manifest: {'ok' if manifest else 'missing'}")
    for name, path in (
        ("prepare", exchange / "prepare_receipt.json"),
        ("self-eval", results / "self_eval" / "self_eval_receipt.json"),
        ("teacher", exchange / "teacher_receipt.json"),
        ("finish", results / "finish_receipt.json"),
    ):
        parts = [receipt["part"] for receipt in merge_receipts(exchange, results, name)
                 if receipt.get("complete")]
        detail = f" (partitions done: {', '.join(parts)}; run merge)" if parts and not path.exists() else ""
        print(f"{name}: {'complete' if path.exists() else 'pending'}{detail}")


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    root = find_root()
    os.chdir(root)
    import rmcq  # noqa: F401 - loads .env before local model libraries

    payload = manifest_payload(args)
    if args.preset == "validation-threshold":
        from rmcq.config import MODELS
        if any(MODELS[key].provider != "hf" for key in split_csv(args.models) + [args.judge_model]):
            raise ValueError("All validation students and the fixed judge must use local HF weights via vLLM")
    if args.stage == "prepare":
        args.data_fingerprints = {}
        for dataset in split_csv(args.datasets):
            for split in ("train", args.eval_split):
                path = root / "data" / "processed" / dataset / f"{split}.jsonl"
                if not path.exists():
                    raise FileNotFoundError(f"Missing {path}. Prepare RACE with python prepare_datasets.py.")
                args.data_fingerprints[f"{dataset}/{split}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        payload["data_fingerprints"] = args.data_fingerprints
        preflight = root / f".run_state/validation_preflight{part_suffix(args.part)}.json"
        if args.preset == "validation-threshold" and preflight.exists():
            args.preflight_report = json.loads(preflight.read_text(encoding="utf-8"))
        if args.preset == "validation-threshold" and args.backend == "vllm":
            import importlib.metadata
            payload["runtime_versions"] = {package: importlib.metadata.version(package)
                for package in ("vllm", "torch", "transformers", "mistral-common")}
    experiment_id = args.experiment_id or json_hash(payload)
    if not re.fullmatch(r"[0-9a-f]{12}", experiment_id):
        raise ValueError("experiment id must be the 12-character hexadecimal id printed by prepare")
    exchange = (root / args.exchange_root / experiment_id).resolve()
    results = (root / args.results_root / experiment_id).resolve()
    manifest_path = exchange / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

    if args.stage == "prepare":
        if args.experiment_id and args.experiment_id != json_hash(payload):
            raise ValueError("--experiment-id does not match the current prepare configuration")
        if args.fresh and exchange.exists():
            shutil.rmtree(exchange)
        exchange.mkdir(parents=True, exist_ok=True)
        payload["experiment_id"] = experiment_id
        report = getattr(args, "preflight_report", None)
        # Both partitions write the same manifest, byte for byte, except for the
        # preflight report: each one only smoke-tested its own models. Keep both,
        # keyed by partition, under the same lock that serializes retrieval.
        with exclusive(exchange / "pairs.lock"):
            existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            if args.part is None:
                if report:
                    payload["preflight_report"] = report
            else:
                reports = dict(existing.get("preflight_reports") or {})
                if report:
                    reports[args.part] = report
                if reports:
                    payload["preflight_reports"] = reports
            save_json(manifest_path, payload)
        if args.write_id:
            args.write_id.parent.mkdir(parents=True, exist_ok=True)
            args.write_id.write_text(experiment_id + "\n", encoding="utf-8")
        print(f"experiment_id: {experiment_id}", flush=True)
        stage_prepare(root, exchange, results, args)
        print(f"commit and push: {display_path(exchange, root)}")
        return

    if not args.experiment_id:
        raise ValueError(f"{args.stage} requires --experiment-id")
    if manifest is None:
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if args.stage == "status":
        stage_status(exchange, results, manifest)
        return
    if args.stage == "merge":
        # merge writes the run-wide receipts, so it cannot require them first.
        stage_merge(exchange, results, args, manifest)
        return
    assert_manifest_compatible(manifest)
    # Later stages consume the frozen configuration, not argparse defaults.
    args.eval_split = manifest["eval_split"]
    args.generation_profile = manifest["generation_profile"]
    args.reflection_temperature = manifest["generation_temperatures"]["student_reflection"]
    args.judge_model = manifest["judge_model"]
    args.backend = manifest["student_backend"]
    args.models = ",".join(manifest["models"])
    args.datasets = ",".join(manifest["datasets"])
    args.teacher_model = manifest["teacher_model"]
    args.teacher_role = manifest.get("teacher_role", "reference")
    args.preset = manifest.get("experiment_preset")
    expected = manifest_payload(args)
    for name in ("generation_policy", "training_answer_max_tokens", "training_answer_retry_max_tokens",
                 "validation_answer_max_tokens", "judge_max_tokens", "judge_retry_max_tokens",
                 "reflection_max_tokens", "reflection_retry_max_tokens", "phi2_stop_sequences"):
        if manifest[name] != expected[name]:
            raise RuntimeError(f"Generation policy {name} changed since prepare; use the matching code revision.")
    from rmcq.config import MODELS
    for model, spec in manifest["model_specs"].items():
        if model not in MODELS or spec != {"repo_id": MODELS[model].repo_id, "extra_kwargs": MODELS[model].extra_kwargs}:
            raise RuntimeError(f"Model configuration changed since prepare: {model}")
    relevant_limits = ["max_model_len", "generation_seed", "dtype", "vllm_deterministic", "vllm_max_num_seqs"] if args.stage in ("finish", "self-eval") else [
        "azure_max_tokens", "azure_reasoning_min_tokens", "azure_reasoning_effort"]
    for name in relevant_limits:
        if payload["runtime_limits"][name] != manifest["runtime_limits"][name]:
            raise RuntimeError(f"Runtime setting {name} differs from frozen manifest; align the server configuration.")
    if args.stage in ("finish", "self-eval") and manifest.get("runtime_versions"):
        import importlib.metadata
        for package, version in manifest["runtime_versions"].items():
            if importlib.metadata.version(package) != version:
                raise RuntimeError(f"GPU package {package} differs from prepare ({version}); use the same environment")
    # A partition evaluates only its own students, so it is gated on its own
    # prepare receipt: neither GPU has to wait at the prepare/self-eval boundary
    # for the other one. The run-wide receipt is written by merge.
    prepare_receipt = exchange / f"prepare_receipt{part_suffix(args.part)}.json"
    if not prepare_receipt.exists() or not json.loads(prepare_receipt.read_text(encoding="utf-8")).get("complete"):
        raise RuntimeError(f"{prepare_receipt.name} is missing or incomplete; resume prepare before evaluating or teaching")
    if args.stage == "self-eval":
        stage_finish(exchange, results, args, manifest)
    elif args.stage == "teacher":
        stage_teacher(exchange, results, args, manifest)
        print(f"commit and push: {display_path(exchange / 'teacher', root)}")
    elif args.stage == "finish":
        teacher_receipt = exchange / "teacher_receipt.json"
        if not teacher_receipt.exists() or not json.loads(teacher_receipt.read_text(encoding="utf-8")).get("complete"):
            raise FileNotFoundError("teacher stage is not complete; pull its artifacts first")
        stage_finish(exchange, results, args, manifest)
    else:
        stage_status(exchange, results, manifest)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
