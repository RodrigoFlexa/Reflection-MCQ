#!/usr/bin/env python
"""Two-server top-1 reflection experiment.

Stages:
  prepare  GPU server: retrieve pairs, answer training sources, self-reflect.
  teacher  Petrobras server: GPT answers/reflects, teaches students, evaluates.
  finish   GPU server: students evaluate with own and teacher reflections.
  status   Any server: report which artifacts are complete.
"""

from __future__ import annotations

import argparse
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
    "deepseek-r1-distill-llama-8b-ollama",
    "llama3.1-8b",
)
DEFAULT_DATASETS = ("aqua", "arc", "logiqa2", "openbookqa")
DEFAULT_TEACHER = "gpt-5-4-petrobras"
PIPELINE_VERSION = "top1-two-server-v4"
ANSWER_TEMPERATURE = 0.0
REFLECTION_TEMPERATURE = 0.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "teacher", "finish", "status"))
    parser.add_argument("--experiment-id", help="Required after prepare; printed by that stage.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--backend", choices=("vllm", "hf", "stub"), default=None)
    parser.add_argument("--teacher-backend", choices=("azure", "stub"), default="azure")
    parser.add_argument("--gpu", default=os.environ.get("RMCQ_NOTEBOOK_GPU", "0"))
    parser.add_argument("--validation-cap", type=int)
    parser.add_argument("--train-cap", type=int, help="Smoke tests only; production must use all training items.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--reflection-temperature", type=float, default=REFLECTION_TEMPERATURE)
    parser.add_argument("--exchange-root", default="experiment_exchange")
    parser.add_argument("--results-root", default="data/results/reflection_top1")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.reflection_temperature <= 2.0:
        parser.error("--reflection-temperature must be between 0.0 and 2.0")
    return args


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
                train_cap: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state: dict[str, Any] = {}
    audit = []
    for dataset in datasets:
        folder = root / "data" / "processed" / dataset
        train = load_jsonl(folder / "train.jsonl")
        validation_path = folder / "validation.jsonl"
        if not validation_path.exists():
            raise FileNotFoundError(f"{dataset} has no validation.jsonl; this protocol uses validation only")
        validation = load_jsonl(validation_path)
        if cap is not None and len(validation) > cap:
            validation = random.Random(42).sample(validation, cap)
        train, train_duplicates = dedupe(train)
        validation, validation_duplicates = dedupe(validation)
        validation_stems = {normalize_stem(item) for item in validation}
        before = len(train)
        train = [item for item in train if normalize_stem(item) not in validation_stems]
        if train_cap is not None and len(train) > train_cap:
            train = random.Random(42).sample(train, train_cap)
        state[dataset] = {"train": train, "validation": validation}
        audit.append({
            "dataset": dataset,
            "train": len(train),
            "validation": len(validation),
            "train_duplicates_removed": train_duplicates,
            "validation_duplicates_removed": validation_duplicates,
            "cross_split_removed": before - len(train),
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
    for dataset, splits in state.items():
        train = splits["train"]
        validation = splits["validation"]
        train_embeddings = model.encode(
            [embedding_text(item) for item in train], batch_size=128,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
        )
        validation_embeddings = model.encode(
            [query_prefix + embedding_text(item) for item in validation], batch_size=128,
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


def training_answer_budget(model_key: str) -> int:
    return 512 if model_key == "phi2" else 768


def validation_answer_budget(model_key: str) -> int:
    return 384 if model_key == "phi2" else 512


def reflection_budget(model_key: str, depth: str) -> int:
    if model_key == "phi2":
        return 512 if depth == "simple" else 768
    return 768 if depth == "simple" else 1024


def cached_generate(backend: Any, path: Path, prompts: dict[str, str], max_tokens: int,
                    batch_size: int, fresh: bool, description: str,
                    temperature: float = ANSWER_TEMPERATURE,
                    allow_unresolved: bool = False) -> dict[str, dict[str, Any]]:
    from rmcq.backends.base import GenParams

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
        hash_input = f"{backend.key}\0{max_tokens}\0{temperature}\0{prompt}"
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
            GenParams(max_new_tokens=max_tokens, temperature=temperature),
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
            retry_tokens = max_tokens + max(128, max_tokens // 2)
            if max_len and hasattr(backend, "tokenizer"):
                overflow = []
                for key, prompt, _digest in retry_batch:
                    prompt_tokens = len(backend.render_token_ids(backend.tokenizer, prompt))
                    if prompt_tokens + retry_tokens > max_len:
                        overflow.append((key, prompt_tokens))
                if overflow:
                    key, prompt_tokens = overflow[0]
                    raise RuntimeError(
                        f"{description}: retry needs {retry_tokens} output tokens but "
                        f"{len(overflow)} prompt(s) would exceed context={max_len}; first "
                        f"key={key!r}, prompt={prompt_tokens}."
                    )
            print(
                f"{description}: retrying {len(retry_batch)} truncated/empty generation(s) "
                f"with max_new_tokens={retry_tokens}", flush=True,
            )
            retried = backend.generate(
                [prompt for _, prompt, _ in retry_batch],
                GenParams(max_new_tokens=retry_tokens, temperature=temperature),
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
            if allow_unresolved:
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
            else:
                first = empty_failed[0]
                raise RuntimeError(
                    f"{description}: empty={len(empty_failed)} after the available token "
                    f"budget; first key={first!r}. "
                    "The partial checkpoint was kept, but it will not be reused as valid output."
                )
    return {key: cached[key] for key in prompts}


def resolve_answers(backend: Any, cache_dir: Path, stage: str, generated: dict[str, dict[str, Any]],
                    items: dict[str, dict[str, Any]], batch_size: int, fresh: bool) -> dict[str, dict[str, Any]]:
    from rmcq.prompts import build_judge_prompt, extract_final_answer, parse_judge_verdict

    results: dict[str, dict[str, Any]] = {}
    judge_prompts = {}
    for key, row in generated.items():
        if row.get("finish_reason") in {"content_filter", "length_exhausted"}:
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
        judged = cached_generate(backend, cache_dir / f"judge_{stage}.jsonl", judge_prompts,
                                  128, batch_size, fresh, f"judge {stage}",
                                  allow_unresolved=True)
        for key, row in judged.items():
            if row.get("finish_reason") in {"length_exhausted", "empty_exhausted"}:
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
    if attempt_row.get("eval_method") in {"content_filter", "length_exhausted"}:
        return f"source_answer_{attempt_row['eval_method']}"
    if "correct" in attempt_row and attempt_row["correct"] is None:
        return f"source_answer_{attempt_row.get('eval_method') or 'unresolved'}"
    status = (reflection_row or {}).get("reflection_status", {}).get(depth)
    if status in {"content_filter", "length_exhausted"}:
        return f"source_reflection_{status}"
    return "source_reflection_unavailable"


def unique_sources(pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {pair["source_uid"]: pair["source_item"] for pair in pairs}


def manifest_payload(args: argparse.Namespace) -> dict[str, Any]:
    from rmcq.config import BACKEND
    from rmcq.prompts import (
        ANSWER_PROMPT, STUDENT_REFLECTION_PROMPTS, TEACHER_REFLECTION_PROMPTS, TRANSFER_PROMPT,
    )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "models": split_csv(args.models), "datasets": split_csv(args.datasets),
        "teacher_model": args.teacher_model, "validation_cap": args.validation_cap,
        "train_cap": args.train_cap,
        "student_backend": args.backend or BACKEND,
        "generation_temperatures": {
            "student_answer": ANSWER_TEMPERATURE,
            "student_judge": ANSWER_TEMPERATURE,
            "student_reflection": args.reflection_temperature,
            "gpt_5_4_petrobras": "provider_default; temperature omitted",
        },
        "training_answer_max_tokens": {
            model: training_answer_budget(model) for model in split_csv(args.models)
        },
        "validation_answer_max_tokens": {
            model: validation_answer_budget(model) for model in split_csv(args.models)
        },
        "reflection_max_tokens": {
            model: {
                depth: reflection_budget(model, depth) for depth in ("simple", "complex")
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
    pair_paths = [exchange / "pairs" / f"{dataset}.jsonl" for dataset in datasets]
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
            state, audit = load_splits(root, datasets, args.validation_cap, args.train_cap)
            pairs = retrieve_top1(state, args.embedding_model, args.embedding_device)
            for dataset in datasets:
                save_jsonl(
                    exchange / "pairs" / f"{dataset}.jsonl",
                    [p for p in pairs if p["dataset"] == dataset],
                )
            save_json(exchange / "retrieval_audit.json", audit)

    sources = unique_sources(pairs)
    content_filter_count = 0
    for model_key in split_csv(args.models):
        model_cache = results / "work" / "prepare" / model_key
        with get_backend(model_key, kind=args.backend) as backend:
            answer_prompts = {uid: build_answer_prompt(item) for uid, item in sources.items()}
            generated = cached_generate(backend, model_cache / "train_answers.jsonl", answer_prompts,
                                        training_answer_budget(model_key), args.batch_size, args.fresh, f"{model_key} training answers")
            verdicts = resolve_answers(backend, model_cache, "train", generated, sources,
                                       args.batch_size, args.fresh)
            reflection_outputs: dict[str, dict[str, dict[str, Any]]] = {}
            for depth in REFLECTION_DEPTHS:
                prompts = {
                    uid: build_reflection_prompt(sources[uid], generated[uid]["text"], verdicts[uid]["correct"], depth, "student")
                    for uid in sources if verdicts[uid]["correct"] is not None
                }
                reflection_outputs[depth] = cached_generate(
                    backend, model_cache / f"self_{depth}.jsonl", prompts,
                    reflection_budget(model_key, depth),
                    args.batch_size, args.fresh, f"{model_key} self reflection {depth}",
                    temperature=args.reflection_temperature,
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
            })
        content_filter_count += sum(
            row["answer_finish_reason"] == "content_filter"
            for row in rows
        ) + sum(
            status == "content_filter"
            for row in rows for status in row["reflection_status"].values()
        )
        save_jsonl(exchange / "students" / model_key / "train.jsonl", rows)
    save_json(exchange / "prepare_receipt.json", {
        "pairs": len(pairs), "unique_training_sources": len(sources),
        "student_models": split_csv(args.models),
        "content_filter_events": content_filter_count, "complete": True,
    })


def load_pairs(exchange: Path, datasets: list[str]) -> list[dict[str, Any]]:
    return [row for dataset in datasets for row in load_jsonl(exchange / "pairs" / f"{dataset}.jsonl")]


def stage_teacher(exchange: Path, results: Path, args: argparse.Namespace, manifest: dict[str, Any]) -> None:
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
        generated = cached_generate(backend, cache_dir / "train_answers.jsonl", prompts, 1024,
                                    args.batch_size, args.fresh, "teacher training answers")
        verdicts = resolve_answers(backend, cache_dir, "train", generated, sources,
                                   args.batch_size, args.fresh)
        self_reflections: dict[str, dict[str, dict[str, Any]]] = {}
        for depth in REFLECTION_DEPTHS:
            reflection_prompts = {
                uid: build_reflection_prompt(sources[uid], generated[uid]["text"], verdicts[uid]["correct"], depth, "student")
                for uid in sources if verdicts[uid]["correct"] is not None
            }
            self_reflections[depth] = cached_generate(
                backend, cache_dir / f"self_{depth}.jsonl", reflection_prompts,
                1024 if depth == "simple" else 2048, args.batch_size, args.fresh,
                f"teacher self reflection {depth}",
                temperature=args.reflection_temperature,
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
                    1024 if depth == "simple" else 2048, args.batch_size, args.fresh,
                    f"teacher reflection for {student_model} {depth}",
                    temperature=args.reflection_temperature,
                )
            teacher_rows_by_model[student_model] = [{
                "dataset": sources[uid]["dataset"], "source_uid": uid,
                "student_model": student_model,
                "reflections": {depth: outputs[depth].get(uid, {}).get("text") for depth in REFLECTION_DEPTHS},
                "reflection_status": reflection_status(outputs, uid),
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
            backend, cache_dir / "validation.jsonl", condition_prompts, 1024,
            args.batch_size, args.fresh, "teacher validation",
        )
        condition_verdicts = resolve_answers(
            backend, cache_dir, "validation", condition_generated, condition_items,
            args.batch_size, args.fresh,
        )

    teacher_train = [{
        "dataset": item["dataset"], "source_uid": uid, "item": item,
        "response": generated[uid]["text"],
        "answer_finish_reason": generated[uid]["finish_reason"], **verdicts[uid],
        "reflections": {depth: self_reflections[depth].get(uid, {}).get("text") for depth in REFLECTION_DEPTHS},
        "reflection_status": reflection_status(self_reflections, uid),
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
            **condition_verdicts[key],
        })
    save_jsonl(exchange / "teacher" / "validation.jsonl", validation_rows)
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

    pairs = load_pairs(exchange, manifest["datasets"])
    student_rows_by_model = {
        model: {row["source_uid"]: row for row in load_jsonl(exchange / "students" / model / "train.jsonl")}
        for model in manifest["models"]
    }
    teacher_rows_by_model = {
        model: {row["source_uid"]: row for row in load_jsonl(exchange / "teacher" / "student_reflections" / f"{model}.jsonl")}
        for model in manifest["models"]
    }
    all_rows = load_jsonl(exchange / "teacher" / "validation.jsonl")
    for model in manifest["models"]:
        prompts: dict[str, str] = {}
        items: dict[str, dict[str, Any]] = {}
        metadata: dict[str, dict[str, Any]] = {}
        unavailable_rows: list[dict[str, Any]] = []
        for pair in pairs:
            val_item, source_item = pair["validation_item"], pair["source_item"]
            source_uid = pair["source_uid"]
            for condition in ("baseline", "self_simple", "self_complex", "teacher_simple", "teacher_complex"):
                key = cache_key(pair["dataset"], pair["val_uid"], condition)
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
                metadata[key] = {"condition": condition, "pair": pair}
        cache_dir = results / "work" / "finish" / model
        with get_backend(model, kind=args.backend) as backend:
            generated = cached_generate(backend, cache_dir / "validation.jsonl", prompts, validation_answer_budget(model),
                                        args.batch_size, args.fresh, f"{model} validation")
            verdicts = resolve_answers(backend, cache_dir, "validation", generated, items,
                                       args.batch_size, args.fresh)
        model_rows = list(unavailable_rows)
        for key, meta in metadata.items():
            pair = meta["pair"]
            model_rows.append({
                "model": model, "dataset": pair["dataset"], "val_uid": pair["val_uid"],
                "source_uid": pair["source_uid"], "similarity": pair["similarity"],
                "condition": meta["condition"], "response": generated[key]["text"],
                "finish_reason": generated[key]["finish_reason"],
                **verdicts[key],
            })
        save_jsonl(results / "models" / model / "validation.jsonl", model_rows)
        all_rows.extend(model_rows)
    save_jsonl(results / "analysis" / "all_outcomes.jsonl", all_rows)
    summary = summarize(all_rows)
    save_csv(results / "analysis" / "accuracy.csv", summary)
    filter_audit = [
        row for row in all_rows if "content_filter" in (row.get("eval_method") or "")
    ]
    save_jsonl(results / "analysis" / "content_filter_audit.jsonl", filter_audit)
    save_json(results / "finish_receipt.json", {
        "rows": len(all_rows), "content_filter_affected_conditions": len(filter_audit),
        "unresolved_conditions": sum(row.get("correct") is None for row in all_rows),
        "complete": True,
    })
    print(f"completed: {results}", flush=True)


def stage_status(exchange: Path, results: Path, manifest: dict[str, Any] | None) -> None:
    print(f"exchange: {exchange}")
    print(f"manifest: {'ok' if manifest else 'missing'}")
    for name, path in (
        ("prepare", exchange / "prepare_receipt.json"),
        ("teacher", exchange / "teacher_receipt.json"),
        ("finish", results / "finish_receipt.json"),
    ):
        print(f"{name}: {'complete' if path.exists() else 'pending'}")


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    root = find_root()
    os.chdir(root)
    import rmcq  # noqa: F401 - loads .env before local model libraries

    payload = manifest_payload(args)
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
        save_json(manifest_path, payload)
        print(f"experiment_id: {experiment_id}", flush=True)
        stage_prepare(root, exchange, results, args)
        print(f"commit and push: {display_path(exchange, root)}")
        return

    if not args.experiment_id:
        raise ValueError(f"{args.stage} requires --experiment-id")
    if manifest is None:
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    assert_manifest_compatible(manifest)
    if args.stage == "teacher":
        stage_teacher(exchange, results, args, manifest)
        print(f"commit and push: {display_path(exchange / 'teacher', root)}")
    elif args.stage == "finish":
        if not (exchange / "teacher_receipt.json").exists():
            raise FileNotFoundError("teacher stage is not complete; pull its artifacts first")
        stage_finish(exchange, results, args, manifest)
    else:
        stage_status(exchange, results, manifest)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
