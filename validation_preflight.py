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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--model", help="Internal: smoke-check just this model in an isolated process")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import rmcq  # loads .env before CUDA
    from rmcq.config import MODELS
    from run_experiment import VALIDATION_MODELS, DEFAULT_DATASETS
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
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for GPU {args.gpu}")
    missing = [str(ROOT / "data/processed" / d / f"{s}.jsonl")
               for d in DEFAULT_DATASETS for s in ("train", "validation")
               if not (ROOT / "data/processed" / d / f"{s}.jsonl").exists()]
    if missing:
        raise FileNotFoundError("Missing canonical datasets:\n" + "\n".join(missing))
    print(f"GPU: {torch.cuda.get_device_name(0)}; versions: {versions}", flush=True)
    for model in VALIDATION_MODELS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--model", model, "--gpu", args.gpu],
                       cwd=ROOT, check=True)
    report = {"complete": True, "versions": versions, "gpu": torch.cuda.get_device_name(0),
              "models": {key: MODELS[key].repo_id for key in VALIDATION_MODELS}}
    path = ROOT / ".run_state/validation_preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("All nine vLLM engines passed. Smoke generations are not experiment outcomes.", flush=True)


if __name__ == "__main__":
    main()
