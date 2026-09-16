#!/usr/bin/env python
"""Download RACE (middle + high) and write the project's canonical MCQ splits."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# A raiz do repositório, não a de tools/, é onde moram rmcq e run_experiment.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_experiment import find_root, save_json, save_jsonl


def normalize_race(row: dict, split: str, index: int) -> dict:
    options = row["options"]
    answer = row["answer"].strip().upper()
    if len(options) != 4 or answer not in "ABCD" or len(answer) != 1:
        raise ValueError(f"Invalid RACE options/answer at {split}:{index}")
    article, question = row["article"].strip(), row["question"].strip()
    if not article or not question:
        raise ValueError(f"Empty RACE article/question at {split}:{index}")
    source = row["example_id"]
    subset = "high" if source.startswith("high") else "middle" if source.startswith("middle") else None
    if subset is None:
        raise ValueError(f"Unknown RACE level: {source}")
    # example_id identifies an article, NOT a question. Several rows share it.
    return {
        "uid": f"race-{split}-{index:06d}", "dataset": "race", "split": split,
        "problem_type": "process", "context": article, "question": question,
        "choices": [{"label": label, "text": text.strip()} for label, text in zip("ABCD", options)],
        "answerKey": answer, "num_choices": 4, "rationale": None,
        "source_id": source, "race_subset": subset,
        "article_uid": hashlib.sha256(" ".join(article.casefold().split()).encode()).hexdigest()[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=["race"], default="race")
    parser.add_argument("--revision", default="main", help="Hub revision; resolved to an immutable commit.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--refresh", action="store_true", help="Download again instead of reusing verified local files.")
    args = parser.parse_args()
    folder = (args.output_root or find_root() / "data" / "processed") / "race"
    manifest_path = folder / "dataset_manifest.json"
    if manifest_path.exists() and not args.refresh:
        import json
        from rmcq.handoff import sha256
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.revision in ("main", existing.get("revision")) and all(
            (folder / f"{split}.jsonl").exists()
            and sha256(folder / f"{split}.jsonl") == existing.get("splits", {}).get(split, {}).get("sha256")
            for split in ("train", "validation", "test")
        ):
            print(f"RACE already installed and verified: {folder} (revision {existing['revision']})")
            return
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info("ehovy/race", revision=args.revision).sha
    dataset = load_dataset("ehovy/race", "all", revision=revision)
    metadata = {"dataset": "ehovy/race", "config": "all", "revision": revision, "splits": {}}
    for split in ("train", "validation", "test"):
        rows = [normalize_race(row, split, i) for i, row in enumerate(dataset[split])]
        path = folder / f"{split}.jsonl"
        save_jsonl(path, rows)
        metadata["splits"][split] = {
            "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "middle": sum(row["race_subset"] == "middle" for row in rows),
            "high": sum(row["race_subset"] == "high" for row in rows),
        }
        print(f"{split}: {len(rows)} questions -> {path}")
    save_json(folder / "dataset_manifest.json", metadata)


if __name__ == "__main__":
    main()
