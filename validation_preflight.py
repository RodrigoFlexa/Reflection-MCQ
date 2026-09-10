#!/usr/bin/env python
"""Check data and run one tiny vLLM generation per validation model before the long job."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
# Failure signatures worth naming, so the log says what to do and not only what
# broke. Each entry is (substrings that must all appear, what to do about it).
KNOWN_FAILURES = (
    (("not subscriptable", "flashinfer"),
     ("FlashInfer annotates array.array[int], which only Python 3.12+ can subscript. "
      "Run python repair_flashinfer_annotations.py in this same venv, then start again.")),
    (("CUDA out of memory",),
     ("The GPU had no room for this engine. Use a free GPU, or lower "
      "RMCQ_VLLM_MAX_NUM_SEQS or the model's max_model_len before starting again.")),
    (("gated repo",),
     "The checkpoint needs accepted terms and a valid HF_TOKEN on this server."),
)


def diagnose(output: str) -> list[str]:
    return [advice for needles, advice in KNOWN_FAILURES
            if all(needle.lower() in output.lower() for needle in needles)]


def models_to_check(part: str | None) -> list[str]:
    """A partition still loads the fixed judge on its own GPU, so it is checked too."""
    from run_experiment import VALIDATION_MODELS, VALIDATION_PARTITIONS, DEFAULT_JUDGE
    if part is None:
        return list(VALIDATION_MODELS)
    if part not in VALIDATION_PARTITIONS:
        raise ValueError(f"Unknown partition {part!r}")
    return [model for model in VALIDATION_MODELS
            if model in VALIDATION_PARTITIONS[part] or model == DEFAULT_JUDGE]


def smoke(model: str, gpu: str, error_log: Path) -> None:
    """Run one model's check in its own process, keeping the whole traceback."""
    print(f"Starting vLLM smoke: {model}", flush=True)
    result = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--model", model, "--gpu", gpu],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="", flush=True)
    if result.returncode == 0:
        return
    advice = diagnose(result.stdout)
    error_log.parent.mkdir(parents=True, exist_ok=True)
    error_log.write_text(json.dumps({"model": model, "gpu": gpu, "returncode": result.returncode,
                                     "advice": advice, "output": result.stdout}, indent=2) + "\n",
                         encoding="utf-8")
    for line in advice:
        print(f"DIAGNOSIS: {line}", flush=True)
    raise RuntimeError(f"vLLM preflight failed for {model}; full output saved in {error_log}"
                       + (f". {advice[0]}" if advice else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--part", help="Check only this partition's models, plus the fixed judge")
    parser.add_argument("--model", help="Internal: smoke-check just this model in an isolated process")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from repair_flashinfer_annotations import check_compatibility
    flashinfer_report = check_compatibility()
    import rmcq  # loads .env before CUDA
    from rmcq.config import MODELS
    from run_experiment import VALIDATION_MODELS, DEFAULT_DATASETS, part_suffix
    from packaging.version import Version
    versions = {p: importlib.metadata.version(p) for p in ("vllm", "mistral-common", "torch", "transformers")}
    for package, minimum in (("vllm", "0.12.0"), ("mistral-common", "1.8.6")):
        if Version(versions[package]) < Version(minimum):
            raise RuntimeError(f"Need {package}>={minimum}; install requirements-validation.txt in the GPU environment")
    if args.model:
        if args.model not in VALIDATION_MODELS:
            raise ValueError("Not a validation model")
        from rmcq.backends import get_backend, GenParams
        with get_backend(args.model, kind="vllm") as backend:
            backend.count_tokens("A small tokenizer check.")
            output = backend.generate(["Reply with the letter A."], GenParams(max_new_tokens=16))
            if len(output) != 1 or output[0].completion_tokens <= 0:
                raise RuntimeError(f"vLLM did not generate tokens for {args.model}")
        print(f"vLLM smoke OK: {args.model}", flush=True)
        return
    checked = models_to_check(args.part)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for GPU {args.gpu}")
    missing = [str(ROOT / "data/processed" / d / f"{s}.jsonl")
               for d in DEFAULT_DATASETS for s in ("train", "validation")
               if not (ROOT / "data/processed" / d / f"{s}.jsonl").exists()]
    if missing:
        raise FileNotFoundError("Missing canonical datasets:\n" + "\n".join(missing))
    suffix = part_suffix(args.part)
    error_log = ROOT / f".run_state/validation_preflight_error{suffix}.json"
    print(f"GPU {args.gpu}: {torch.cuda.get_device_name(0)}; versions: {versions}", flush=True)
    print(f"Checking {len(checked)} engines"
          f"{f' for partition {args.part}' if args.part else ''}: {', '.join(checked)}", flush=True)
    for model in checked:
        smoke(model, args.gpu, error_log)
    report = {"complete": True, "part": args.part, "versions": versions,
              "gpu": torch.cuda.get_device_name(0), "gpu_index": args.gpu,
              "flashinfer": flashinfer_report,
              "models": {key: MODELS[key].repo_id for key in checked}}
    path = ROOT / f".run_state/validation_preflight{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    error_log.unlink(missing_ok=True)
    print(f"All {len(checked)} vLLM engines passed. Smoke generations are not experiment outcomes.", flush=True)


if __name__ == "__main__":
    main()
