#!/usr/bin/env python
"""Add an Azure-generated reflection condition to a completed experiment 08.

The workflow intentionally separates machines:

1. ``export`` (GPU server) creates Git-friendly reflection requests.
2. ``generate`` (Petrobras server) calls the Azure deployment and checkpoints replies.
3. ``finish`` (GPU server) evaluates those replies with each original student and
   rebuilds the threshold analysis with ``depth=external_reflection``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


EXTERNAL_DEPTH = "external_reflection"
DEFAULT_RESULTS_SUBDIR = Path("data/results/similarity_threshold_v2")
DEFAULT_EXCHANGE_SUBDIR = Path("external_reflection_exchange")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "generate", "finish", "status"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--reflection-model", default="gpt-5-4-petrobras")
    parser.add_argument("--results-root", help="Root containing experiment directories.")
    parser.add_argument("--exchange-root", help="Git-tracked exchange directory.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", default=os.environ.get("RMCQ_NOTEBOOK_GPU", "3"))
    parser.add_argument("--student-backend", default="vllm", choices=("vllm", "hf", "stub"))
    parser.add_argument("--reflection-backend", default="azure", choices=("azure", "stub"))
    parser.add_argument("--fresh", action="store_true", help="Ignore only this stage's caches.")
    return parser.parse_args()


def find_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "rmcq").is_dir():
            return candidate.resolve()
    raise RuntimeError("Repository root containing rmcq/ was not found.")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def cache_key(*parts: Any) -> str:
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_manifest(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != experiment_dir.name:
        raise RuntimeError("Manifest experiment_id does not match its directory name.")
    return manifest


def format_options(choices: list[dict[str, Any]]) -> str:
    return "\n".join(f"{choice['label']}) {choice['text']}" for choice in choices)


def format_question(item: dict[str, Any]) -> str:
    context = (item.get("context") or "").strip()
    question = item["question"].strip()
    return f"{context}\n\n{question}" if context else question


def build_reflection_prompt(
    instruction: str,
    item: dict[str, Any],
    previous_answer: str,
    was_correct: bool,
) -> str:
    correct_label = item["answerKey"]
    correct_text = next(c["text"] for c in item["choices"] if c["label"] == correct_label)
    outcome = "CORRECT" if was_correct else "INCORRECT"
    return (
        f"{instruction}\n\n"
        f"Problem:\n{format_question(item)}\n\n"
        f"Options:\n{format_options(item['choices'])}\n\n"
        f"Your previous response:\n{previous_answer.strip()}\n\n"
        f"Outcome: {outcome}\n"
        f"Correct answer for private feedback: {correct_label}) {correct_text}"
    )


def build_memory_prompt(template: str, item: dict[str, Any], memory: str) -> str:
    if template.count("Memory:") != 1:
        raise RuntimeError("The experiment memory prompt must contain exactly one Memory: marker.")
    return template.format(
        memory=(memory or "").strip(),
        question=format_question(item),
        options=format_options(item["choices"]),
    )


JUDGE_PROMPT = """You are grading a multiple-choice response.

Question: {question}

Options:
{options}

Correct option: {correct_label}) {correct_text}

Candidate response:
{response}

Decide only which option the candidate ultimately selected.
End with this exact line and nothing after it:
Verdict: <CORRECT or INCORRECT>"""


def build_judge_prompt(item: dict[str, Any], response: str) -> str:
    label = item["answerKey"]
    correct_text = next(c["text"] for c in item["choices"] if c["label"] == label)
    return JUDGE_PROMPT.format(
        question=format_question(item),
        options=format_options(item["choices"]),
        correct_label=label,
        correct_text=correct_text,
        response=(response or "").strip(),
    )


def load_item_maps(root: Path, datasets: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in datasets:
        merged: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation"):
            path = root / "data" / "processed" / dataset / f"{split}.jsonl"
            if path.exists():
                merged.update({row["uid"]: row for row in load_jsonl(path)})
        maps[dataset] = merged
    return maps


def exchange_paths(exchange_dir: Path, student: str, dataset: str) -> tuple[Path, Path]:
    filename = safe_name(dataset) + ".jsonl"
    student_dir = safe_name(student)
    return (
        exchange_dir / "requests" / student_dir / filename,
        exchange_dir / "responses" / student_dir / filename,
    )


def export_requests(root: Path, experiment_dir: Path, exchange_dir: Path, args: argparse.Namespace) -> None:
    import pandas as pd

    manifest = read_manifest(experiment_dir)
    models = list(manifest["models"])
    datasets = list(manifest["datasets"])
    prompts = manifest.get("reflection_prompts", {})
    if "diagnostic" not in prompts:
        raise RuntimeError("This experiment has no diagnostic reflection prompt to match externally.")
    instruction = prompts["diagnostic"]
    item_maps = load_item_maps(root, datasets)
    outcomes_path = experiment_dir / "analysis" / "all_outcomes.csv"
    if not outcomes_path.exists():
        raise FileNotFoundError("Run experiment 08 through completion before exporting external reflections.")
    outcomes = pd.read_csv(outcomes_path, low_memory=False)

    counts: dict[str, dict[str, int]] = {}
    for student in models:
        source_path = experiment_dir / "generations" / student / "source_answers.jsonl"
        source_rows = load_jsonl(source_path)
        correctness = (
            outcomes[outcomes.model == student]
            .dropna(subset=["source_correct"])
            .drop_duplicates(["dataset", "source_uid"])
            .set_index(["dataset", "source_uid"])["source_correct"]
            .to_dict()
        )
        requests_by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in datasets}
        for source in source_rows:
            parts = json.loads(source["key"])
            if len(parts) != 3 or parts[0] != "source":
                continue
            _, dataset, uid = parts
            was_correct = correctness.get((dataset, uid))
            item = item_maps.get(dataset, {}).get(uid)
            if item is None or was_correct is None:
                continue
            prompt = build_reflection_prompt(instruction, item, source["text"], bool(was_correct))
            key = cache_key("reflection", EXTERNAL_DEPTH, dataset, uid)
            requests_by_dataset[dataset].append({
                "key": key,
                "experiment_id": args.experiment_id,
                "student_model": student,
                "reflection_model": args.reflection_model,
                "dataset": dataset,
                "source_uid": uid,
                "prompt_sha256": text_hash(prompt),
                "prompt": prompt,
            })
        counts[student] = {}
        for dataset in datasets:
            request_path, _ = exchange_paths(exchange_dir, student, dataset)
            save_jsonl(request_path, requests_by_dataset[dataset])
            counts[student][dataset] = len(requests_by_dataset[dataset])
            print(
                f"{student}/{dataset}: exported {len(requests_by_dataset[dataset]):,} "
                f"requests -> {request_path}", flush=True,
            )

    exchange_manifest = {
        "format_version": 1,
        "experiment_id": args.experiment_id,
        "reflection_model": args.reflection_model,
        "reflection_depth": EXTERNAL_DEPTH,
        "prompt_reference": "diagnostic (same instruction; generator is the experimental change)",
        "instruction_sha256": text_hash(instruction),
        "students": models,
        "datasets": datasets,
        "request_counts": counts,
    }
    save_json(exchange_dir / "exchange_manifest.json", exchange_manifest)
    print("Export complete. Commit exchange_manifest.json and requests/ to Git.", flush=True)


def generate_reflections(exchange_dir: Path, args: argparse.Namespace) -> None:
    import rmcq  # noqa: F401 - loads .env before backend configuration
    from rmcq.backends import get_backend
    from rmcq.backends.base import GenParams

    exchange_manifest = json.loads((exchange_dir / "exchange_manifest.json").read_text(encoding="utf-8"))
    if exchange_manifest["experiment_id"] != args.experiment_id:
        raise RuntimeError("Exchange belongs to another experiment.")
    if exchange_manifest["reflection_model"] != args.reflection_model:
        raise RuntimeError("Exchange belongs to another reflection model.")

    counts: dict[str, dict[str, dict[str, int]]] = {}
    with get_backend(args.reflection_model, kind=args.reflection_backend) as backend:
        for student in exchange_manifest["students"]:
            counts[student] = {}
            for dataset in exchange_manifest["datasets"]:
                request_path, response_path = exchange_paths(exchange_dir, student, dataset)
                requests = load_jsonl(request_path)
                existing_rows = [] if args.fresh or not response_path.exists() else load_jsonl(response_path)
                cache = {row["key"]: row for row in existing_rows}
                missing = [
                    row for row in requests
                    if row["key"] not in cache
                    or cache[row["key"]].get("prompt_sha256") != row["prompt_sha256"]
                ]
                print(
                    f"{student}/{dataset}: total={len(requests):,} "
                    f"cache={len(requests)-len(missing):,} missing={len(missing):,}", flush=True,
                )
                for start in range(0, len(missing), args.batch_size):
                    batch = missing[start:start + args.batch_size]
                    generated = backend.generate(
                        [row["prompt"] for row in batch],
                        GenParams(max_new_tokens=180),
                        desc=(f"external reflections {student}/{dataset} "
                              f"[{start + 1}-{start + len(batch)}]"),
                    )
                    for request, generation in zip(batch, generated):
                        cache[request["key"]] = {
                            "key": request["key"],
                            "experiment_id": args.experiment_id,
                            "student_model": student,
                            "reflection_model": args.reflection_model,
                            "prompt_sha256": request["prompt_sha256"],
                            "text": generation.text,
                            "prompt_tokens": generation.prompt_tokens,
                            "completion_tokens": generation.completion_tokens,
                            "finish_reason": generation.finish_reason,
                        }
                    ordered = [cache[row["key"]] for row in requests if row["key"] in cache]
                    save_jsonl(response_path, ordered)
                ordered = [
                    cache[row["key"]] for row in requests
                    if row["key"] in cache
                    and cache[row["key"]].get("prompt_sha256") == row["prompt_sha256"]
                ]
                save_jsonl(response_path, ordered)
                counts[student][dataset] = {"requested": len(requests), "completed": len(ordered)}

    complete = all(
        value["requested"] == value["completed"]
        for student_counts in counts.values() for value in student_counts.values()
    )
    receipt = {
        "format_version": 1,
        "experiment_id": args.experiment_id,
        "reflection_model": args.reflection_model,
        "counts": counts,
        "complete": complete,
    }
    save_json(exchange_dir / "generation_receipt.json", receipt)
    if not complete:
        raise RuntimeError("Generation is incomplete; rerun with the same command to resume.")
    print("Azure generation complete. Commit responses/ and generation_receipt.json to Git.", flush=True)


def cached_generate(backend: Any, path: Path, prompts: dict[str, str], max_tokens: int,
                    batch_size: int, fresh: bool, desc: str) -> dict[str, str]:
    from rmcq.backends.base import GenParams

    cache = {} if fresh or not path.exists() else {row["key"]: row for row in load_jsonl(path)}
    missing = [
        (key, prompt) for key, prompt in prompts.items()
        if key not in cache or cache[key].get("prompt_sha256") != text_hash(prompt)
    ]
    print(f"{desc}: total={len(prompts):,} cache={len(prompts)-len(missing):,} missing={len(missing):,}")
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        generated = backend.generate(
            [prompt for _, prompt in batch], GenParams(max_new_tokens=max_tokens),
            desc=f"{desc} [{start + 1}-{start + len(batch)}]",
        )
        for (key, prompt), generation in zip(batch, generated):
            cache[key] = {
                "key": key,
                "prompt_sha256": text_hash(prompt),
                "text": generation.text,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "finish_reason": generation.finish_reason,
            }
        save_jsonl(path, cache.values())
    return {key: cache[key]["text"] for key in prompts}


def resolve_correctness(backend: Any, path: Path, texts: dict[str, str],
                        items: dict[str, dict[str, Any]], batch_size: int,
                        fresh: bool) -> tuple[dict[str, bool | None], dict[str, str]]:
    from rmcq.thresholds import extract_final_answer

    correctness: dict[str, bool | None] = {}
    method: dict[str, str] = {}
    fallback: dict[str, str] = {}
    for key, text in texts.items():
        answer = extract_final_answer(text)
        if answer is None:
            fallback[key] = build_judge_prompt(items[key], text)
        else:
            correctness[key] = answer == items[key]["answerKey"]
            method[key] = "parser"
    if fallback:
        judged = cached_generate(backend, path, fallback, 80, batch_size, fresh, "external judge fallback")
        for key, text in judged.items():
            match = re.search(r"Verdict:\s*(CORRECT|INCORRECT)", text or "", re.I)
            correctness[key] = None if not match else match.group(1).upper() == "CORRECT"
            method[key] = "judge_fallback" if correctness[key] is not None else "unresolved"
    return correctness, method


def pool_mask(frame: Any, pool: str) -> Any:
    if pool == "errors":
        return frame["source_correct"] == False  # noqa: E712
    if pool == "correct":
        return frame["source_correct"] == True  # noqa: E712
    return frame["source_correct"].notna()


def rebuild_analysis(experiment_dir: Path, manifest: dict[str, Any]) -> None:
    import numpy as np
    import pandas as pd
    from scipy.stats import binomtest
    from rmcq.thresholds import (
        clustered_bootstrap_curve,
        paired_bootstrap_difference,
        sustained_threshold,
        threshold_bootstrap_ci,
    )

    results = pd.read_csv(experiment_dir / "analysis" / "all_outcomes.csv", low_memory=False)
    pools = ["all", "errors", "correct"]
    bandwidth = float(manifest["kernel_bandwidth"])
    n_curve = int(manifest["bootstrap_curve"])
    n_policy = int(manifest["bootstrap_policy"])
    min_n = int(manifest["min_effective_n"])
    consecutive = int(manifest["sustained_grid_points"])
    seed = int(manifest["seed"])
    curve_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for (model, dataset, depth), group in results.groupby(["model", "dataset", "depth"]):
        calibration = group[
            (group.analysis_split == "calibration") & (group.arm == "retrieved")
            & group.delta.notna()
        ].copy()
        for pool in pools:
            sub = calibration[pool_mask(calibration, pool)].copy()
            clusters = sub.val_uid.nunique()
            base = {
                "model": model, "dataset": dataset, "depth": depth, "pool": pool,
                "n_rows_calibration": len(sub), "n_questions_calibration": clusters,
            }
            if len(sub) < 40 or clusters < 20 or sub.similarity.nunique() < 10:
                threshold_rows.append(base | {
                    "threshold": None, "threshold_ci_low": None, "threshold_ci_high": None,
                    "threshold_identification_rate": 0.0, "confident_help_threshold": None,
                    "confident_harm_region_start": None, "status": "insufficient_data",
                })
                continue
            lo, hi = np.quantile(sub.similarity, [0.02, 0.98])
            grid = np.linspace(lo, hi, 81)
            records = sub[["val_uid", "similarity", "delta"]].to_dict("records")
            curve = clustered_bootstrap_curve(
                records, grid, bandwidth=bandwidth, n_boot=n_curve,
                seed=seed + sum(ord(c) for c in f"{model}{dataset}{depth}{pool}"),
            )
            threshold = sustained_threshold(
                grid, curve["estimate"], curve["effective_n"],
                min_effective_n=min_n, consecutive=consecutive,
            )
            ci_low, ci_high, identification_rate = threshold_bootstrap_ci(
                grid, curve["bootstrap"], curve["effective_n"],
                min_effective_n=min_n, consecutive=consecutive,
            )
            confident_help = sustained_threshold(
                grid, curve["low"], curve["effective_n"],
                min_effective_n=min_n, consecutive=consecutive,
            )
            confident_harm = sustained_threshold(
                grid, -curve["high"], curve["effective_n"],
                min_effective_n=min_n, consecutive=consecutive,
            )
            threshold_rows.append(base | {
                "threshold": threshold, "threshold_ci_low": ci_low, "threshold_ci_high": ci_high,
                "threshold_identification_rate": identification_rate,
                "confident_help_threshold": confident_help,
                "confident_harm_region_start": confident_harm,
                "status": "identified" if threshold is not None else "no_crossing",
            })
            for index, similarity in enumerate(grid):
                curve_rows.append({
                    "model": model, "dataset": dataset, "depth": depth, "pool": pool,
                    "similarity": similarity, "effect": curve["estimate"][index],
                    "ci_low": curve["low"][index], "ci_high": curve["high"][index],
                    "effective_n": curve["effective_n"][index],
                })

    threshold_frame = pd.DataFrame(threshold_rows)
    pd.DataFrame(curve_rows).to_csv(experiment_dir / "analysis" / "effect_curves_calibration.csv", index=False)
    threshold_frame.to_csv(experiment_dir / "analysis" / "thresholds_calibration.csv", index=False)

    def with_mcnemar(result: dict[str, Any]) -> dict[str, Any]:
        discordant = result["helped"] + result["harmed"]
        result["mcnemar_p"] = float(binomtest(result["helped"], discordant, 0.5).pvalue) if discordant else 1.0
        return result

    policy_rows: list[dict[str, Any]] = []
    for row in threshold_rows:
        model, dataset, depth, pool = row["model"], row["dataset"], row["depth"], row["pool"]
        group = results[
            (results.model == model) & (results.dataset == dataset) & (results.depth == depth)
            & (results.analysis_split == "test")
        ].copy()
        top = group[(group.arm == "retrieved") & (group.level == "top")].copy()
        top = top[top.baseline_correct.notna() & top.memory_correct.notna()]
        if top.empty:
            continue
        eligible = pool_mask(top, pool)
        always = np.where(eligible, top.memory_correct, top.baseline_correct).astype(float)
        always_result = with_mcnemar(paired_bootstrap_difference(
            top.baseline_correct.astype(float), always, n_boot=n_policy, seed=seed,
        ))
        policy_rows.append({
            "model": model, "dataset": dataset, "depth": depth, "pool": pool,
            "policy": "always_top", "threshold": None,
        } | always_result)
        if row["threshold"] is not None:
            use = eligible & (top.similarity >= row["threshold"])
            policy = np.where(use, top.memory_correct, top.baseline_correct).astype(float)
            policy_result = with_mcnemar(paired_bootstrap_difference(
                top.baseline_correct.astype(float), policy, n_boot=n_policy, seed=seed + 1,
            ))
            policy_rows.append({
                "model": model, "dataset": dataset, "depth": depth, "pool": pool,
                "policy": "threshold_top", "threshold": row["threshold"],
                "memory_use_rate": float(use.mean()),
            } | policy_result)
        if pool == "all":
            placebo = group[(group.arm == "placebo") & group.baseline_correct.notna() & group.memory_correct.notna()]
            if not placebo.empty:
                placebo_result = with_mcnemar(paired_bootstrap_difference(
                    placebo.baseline_correct.astype(float), placebo.memory_correct.astype(float),
                    n_boot=n_policy, seed=seed + 2,
                ))
                policy_rows.append({
                    "model": model, "dataset": dataset, "depth": depth, "pool": pool,
                    "policy": "placebo", "threshold": None,
                } | placebo_result)

    pd.DataFrame(policy_rows).to_csv(experiment_dir / "analysis" / "heldout_policies.csv", index=False)
    transitions = results[results.delta.notna()].copy()
    transitions["transition"] = np.select(
        [transitions.delta > 0, transitions.delta < 0], ["helped", "harmed"], default="unchanged"
    )
    transition_frame = (
        transitions.groupby(["model", "dataset", "depth", "arm", "level", "transition"])
        .size().rename("n").reset_index()
    )
    transition_frame.to_csv(experiment_dir / "analysis" / "decision_transitions.csv", index=False)


def finish_experiment(root: Path, experiment_dir: Path, exchange_dir: Path,
                      args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    import rmcq  # noqa: F401
    from rmcq.backends import get_backend

    manifest = read_manifest(experiment_dir)
    receipt_path = exchange_dir / "generation_receipt.json"
    if not receipt_path.exists() or not json.loads(receipt_path.read_text(encoding="utf-8")).get("complete"):
        raise RuntimeError("External generation receipt is absent or incomplete.")
    exchange_manifest = json.loads((exchange_dir / "exchange_manifest.json").read_text(encoding="utf-8"))
    if exchange_manifest["experiment_id"] != args.experiment_id:
        raise RuntimeError("Exchange belongs to another experiment.")

    datasets = list(manifest["datasets"])
    item_maps = load_item_maps(root, datasets)
    pairs = pd.read_csv(experiment_dir / "retrieval" / "pairs.csv", low_memory=False)
    existing = pd.read_csv(experiment_dir / "analysis" / "all_outcomes.csv", low_memory=False)
    existing = existing[existing.depth != EXTERNAL_DEPTH].copy()
    memory_template = manifest["memory_prompt"]
    external_rows: list[dict[str, Any]] = []

    for student in manifest["models"]:
        print(f"\nSTUDENT {student} | reflection {args.reflection_model}", flush=True)
        request_rows: list[dict[str, Any]] = []
        response_rows: list[dict[str, Any]] = []
        for dataset in datasets:
            request_path, response_path = exchange_paths(exchange_dir, student, dataset)
            request_rows.extend(load_jsonl(request_path))
            response_rows.extend(load_jsonl(response_path))
        requests = {row["key"]: row for row in request_rows}
        responses = {row["key"]: row for row in response_rows}
        missing = [key for key in requests if key not in responses]
        mismatched = [
            key for key in requests if key in responses
            and responses[key].get("prompt_sha256") != requests[key]["prompt_sha256"]
        ]
        if missing or mismatched:
            raise RuntimeError(f"{student}: incomplete/mismatched exchange ({len(missing)=}, {len(mismatched)=}).")

        imported = [responses[key] for key in requests]
        reflection_cache = experiment_dir / "generations" / student / f"reflections_{EXTERNAL_DEPTH}.jsonl"
        save_jsonl(reflection_cache, imported)
        memories = {}
        for row in imported:
            _, _, dataset, uid = json.loads(row["key"])
            memories[(dataset, uid)] = row["text"]

        old_student = existing[existing.model == student]
        baseline = (
            old_student.dropna(subset=["baseline_correct"])
            .drop_duplicates(["dataset", "val_uid"])
            .set_index(["dataset", "val_uid"])
        )
        source = (
            old_student.dropna(subset=["source_correct"])
            .drop_duplicates(["dataset", "source_uid"])
            .set_index(["dataset", "source_uid"])
        )
        prompts: dict[str, str] = {}
        items: dict[str, dict[str, Any]] = {}
        pair_by_key: dict[str, dict[str, Any]] = {}
        for pair in pairs.to_dict("records"):
            memory = memories.get((pair["dataset"], pair["source_uid"]))
            item = item_maps.get(pair["dataset"], {}).get(pair["val_uid"])
            if memory is None or item is None:
                continue
            key = cache_key(
                "memory", pair["dataset"], pair["val_uid"], EXTERNAL_DEPTH,
                pair["arm"], pair["level"], pair["source_uid"],
            )
            prompts[key] = build_memory_prompt(memory_template, item, memory)
            items[key] = item
            pair_by_key[key] = pair

        model_dir = experiment_dir / "generations" / student
        with get_backend(student, kind=args.student_backend) as backend:
            answers = cached_generate(
                backend, model_dir / f"validation_with_memory_{EXTERNAL_DEPTH}.jsonl",
                prompts, 400, args.batch_size, args.fresh, "validation with external reflection",
            )
            correct, method = resolve_correctness(
                backend, model_dir / f"judge_fallback_memory_{EXTERNAL_DEPTH}.jsonl",
                answers, items, args.batch_size, args.fresh,
            )

        for key, pair in pair_by_key.items():
            baseline_index = (pair["dataset"], pair["val_uid"])
            source_index = (pair["dataset"], pair["source_uid"])
            b = None if baseline_index not in baseline.index else bool(baseline.loc[baseline_index, "baseline_correct"])
            s = None if source_index not in source.index else bool(source.loc[source_index, "source_correct"])
            m = correct.get(key)
            external_rows.append({
                "model": student,
                "reflection_model": args.reflection_model,
                "dataset": pair["dataset"], "val_uid": pair["val_uid"],
                "source_uid": pair["source_uid"], "analysis_split": pair["analysis_split"],
                "depth": EXTERNAL_DEPTH, "arm": pair["arm"], "level": pair["level"],
                "requested_similarity": pair["requested_similarity"], "similarity": pair["similarity"],
                "source_correct": s, "baseline_correct": b, "memory_correct": m,
                "delta": None if b is None or m is None else int(m) - int(b),
                "baseline_eval_method": None if baseline_index not in baseline.index else baseline.loc[baseline_index].get("baseline_eval_method"),
                "memory_eval_method": method.get(key),
                "source_eval_method": None if source_index not in source.index else source.loc[source_index].get("source_eval_method"),
            })

        student_rows = pd.concat(
            [old_student, pd.DataFrame([r for r in external_rows if r["model"] == student])],
            ignore_index=True,
        )
        student_rows.to_csv(model_dir / "outcomes.csv", index=False)
        save_jsonl(model_dir / "outcomes.jsonl", student_rows.replace({np.nan: None}).to_dict("records"))

    combined = pd.concat([existing, pd.DataFrame(external_rows)], ignore_index=True)
    combined.to_csv(experiment_dir / "analysis" / "all_outcomes.csv", index=False)
    rebuild_analysis(experiment_dir, manifest)
    manifest["analysis_depths"] = list(dict.fromkeys(list(manifest.get("depths", [])) + [EXTERNAL_DEPTH]))
    manifest["external_reflection"] = {
        "reflection_model": args.reflection_model,
        "prompt_reference": "diagnostic",
        "exchange_format_version": exchange_manifest["format_version"],
        "exchange_relative_path": str(exchange_dir.relative_to(root)) if exchange_dir.is_relative_to(root) else str(exchange_dir),
    }
    save_json(experiment_dir / "manifest.json", manifest)
    print(f"External condition complete: {len(external_rows):,} outcome rows.", flush=True)
    print("Run notebooks/08_similarity_reflection_threshold_plots.ipynb next.", flush=True)


def show_status(experiment_dir: Path, exchange_dir: Path, args: argparse.Namespace) -> None:
    print("experiment:", experiment_dir)
    print("exchange:", exchange_dir)
    for name in ("exchange_manifest.json", "generation_receipt.json"):
        path = exchange_dir / name
        print(name, "ok" if path.exists() else "missing")
    if (exchange_dir / "exchange_manifest.json").exists():
        manifest = json.loads((exchange_dir / "exchange_manifest.json").read_text(encoding="utf-8"))
        for student in manifest["students"]:
            for dataset in manifest["datasets"]:
                request_path, response_path = exchange_paths(exchange_dir, student, dataset)
                requested = len(load_jsonl(request_path)) if request_path.exists() else 0
                completed = len(load_jsonl(response_path)) if response_path.exists() else 0
                print(f"{student}/{dataset}: requests={requested:,} responses={completed:,}")
    result_manifest = read_manifest(experiment_dir)
    print("analysis_depths:", result_manifest.get("analysis_depths", result_manifest.get("depths")))


def main() -> None:
    args = parse_args()
    root = find_root()
    os.environ["RMCQ_NOTEBOOK_GPU"] = str(args.gpu)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    results_root = Path(args.results_root).expanduser().resolve() if args.results_root else root / DEFAULT_RESULTS_SUBDIR
    exchange_root = Path(args.exchange_root).expanduser().resolve() if args.exchange_root else root / DEFAULT_EXCHANGE_SUBDIR
    experiment_dir = results_root / args.experiment_id
    exchange_dir = exchange_root / args.experiment_id / safe_name(args.reflection_model)

    if args.stage == "export":
        export_requests(root, experiment_dir, exchange_dir, args)
    elif args.stage == "generate":
        generate_reflections(exchange_dir, args)
    elif args.stage == "finish":
        finish_experiment(root, experiment_dir, exchange_dir, args)
    else:
        show_status(experiment_dir, exchange_dir, args)


if __name__ == "__main__":
    main()
