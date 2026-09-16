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
import time
import traceback

ROOT = Path(__file__).resolve().parent
# Failure signatures worth naming, so the log says what to do and not only what
# broke. Each entry is (substrings that must all appear, what to do about it).
KNOWN_FAILURES = (
    (("not subscriptable", "flashinfer"),
     ("FlashInfer annotates array.array[int], which only Python 3.12+ can subscript. "
      "Run python tools/repair_flashinfer_annotations.py in this same venv, then start again.")),
    (("CUDA out of memory",),
     ("The GPU had no room for this engine. Use a free GPU, or lower "
      "RMCQ_VLLM_MAX_NUM_SEQS or the model's max_model_len before starting again.")),
    (("fd_exchange annotations are incompatible",),
     ("The preflight guard caught the FlashInfer annotation defect before loading weights. "
      "Run python tools/repair_flashinfer_annotations.py in this same venv, then start again.")),
    (("gated repo",),
     ("The checkpoint needs accepted terms and a valid HF_TOKEN on this server. "
      "Access is per repository: a grant on one Llama repo does not cover another.")),
)


class PreflightFailure(RuntimeError):
    """A failure already written to the error file, with its own diagnosis."""


def diagnose(output: str) -> list[str]:
    return [advice for needles, advice in KNOWN_FAILURES
            if all(needle.lower() in output.lower() for needle in needles)]


def record_failure(error_log: Path, output: str, **fields) -> list[str]:
    advice = diagnose(output)
    error_log.parent.mkdir(parents=True, exist_ok=True)
    error_log.write_text(json.dumps({**fields, "advice": advice, "output": output}, indent=2) + "\n",
                         encoding="utf-8")
    for line in advice:
        print(f"DIAGNOSIS: {line}", flush=True)
    return advice


def check_repo_access(models: list[str], skip_gated: bool = False) -> dict:
    """Ask the Hub which repos this token cannot read, before loading any weights.

    Gating is per repository, and a grant on one Llama repo says nothing about
    another. Checking every repo up front costs a few seconds and reports all of
    them at once, instead of failing after minutes of downloading an earlier
    model. Anything that is not an answer about permission (offline, network,
    rate limit) is only reported: the smoke test remains the real check, and a
    server with everything already cached must still be able to run.

    With `skip_gated` the blocked models are reported and left out instead of
    stopping the run. They stay absent from the results until a later run finds
    them readable, and merge keeps refusing to close the run while any is missing.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    from rmcq.config import MODELS, hf_token
    api = HfApi(token=hf_token())
    blocked, undetermined = [], {}
    for key in models:
        repo = MODELS[key].repo_id
        try:
            api.model_info(repo)
        except (GatedRepoError, RepositoryNotFoundError):
            blocked.append((key, repo))
        except Exception as exc:  # offline, proxy, rate limit: not a permission answer
            undetermined[key] = f"{type(exc).__name__}: {exc}"
    report = {"checked": len(models), "blocked": [repo for _, repo in blocked],
              "blocked_models": [key for key, _ in blocked],
              "undetermined": undetermined, "authenticated": bool(hf_token())}
    for key, reason in undetermined.items():
        print(f"NOTE: could not confirm Hub access for {key}: {reason}", flush=True)
    if blocked and skip_gated:
        for key, repo in blocked:
            print(f"SKIPPING {key}: no access to https://huggingface.co/{repo}. "
                  f"Request it, then run the same command again to fill this gap.", flush=True)
        return report
    if blocked:
        listing = "\n".join(f"  {key}: https://huggingface.co/{repo}" for key, repo in blocked)
        raise RuntimeError(
            f"Cannot read {len(blocked)} gated repo(s) with the token on this server "
            "(gated, private or renamed). Access is granted per repository, so a grant "
            "on one Llama repo does not cover another. Request access on each page below "
            f"with the same Hugging Face account that issued HF_TOKEN, then start again:\n{listing}")
    return report


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
    advice = record_failure(error_log, result.stdout, stage="model_check", model=model,
                            gpu=gpu, returncode=result.returncode)
    raise PreflightFailure(f"vLLM preflight failed for {model}; full output saved in {error_log}"
                           + (f". {advice[0]}" if advice else ""))


def error_log_path(part: str | None) -> Path:
    return ROOT / f".run_state/validation_preflight_error{'' if part is None else '.' + part}.json"


def skips_path(root: Path, part: str | None) -> Path:
    return root / f".run_state/validation_gated_skips{'' if part is None else '.' + part}.json"


def write_skips(part: str | None, skipped: list[str], access: dict) -> None:
    """Record which models this run leaves out, for the generation stages to read.

    Always rewritten, including as an empty list: once access is granted, the
    same command re-checks the Hub, finds nothing blocked, and the stages stop
    skipping without anyone having to remember to clear a file.
    """
    path = skips_path(ROOT, part)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"part": part, "skipped": skipped,
                                "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "hub_access": access}, indent=2) + "\n", encoding="utf-8")


def run(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    from repair_flashinfer_annotations import check_compatibility
    flashinfer_report = check_compatibility()
    import rmcq  # loads .env before CUDA
    from rmcq.config import MODELS
    from run_experiment import VALIDATION_MODELS, DEFAULT_DATASETS, part_suffix
    from packaging.version import Version
    versions = {p: importlib.metadata.version(p) for p in ("vllm", "mistral-common", "torch", "transformers")}
    for package, minimum in (("vllm", "0.12.0"), ("mistral-common", "1.8.6")):
        if Version(versions[package]) < Version(minimum):
            raise RuntimeError(f"Need {package}>={minimum}; install requirements/validation.txt in the GPU environment")
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
    access = check_repo_access(checked, skip_gated=args.skip_gated)
    skipped = access["blocked_models"] if args.skip_gated else []
    # The skip list is what the generation stages read: the decision about which
    # checkpoints are reachable is made once, here, and never guessed from an
    # exception raised halfway through loading a model.
    write_skips(args.part, skipped, access)
    checked = [model for model in checked if model not in skipped]
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for GPU {args.gpu}")
    missing = [str(ROOT / "data/processed" / d / f"{s}.jsonl")
               for d in DEFAULT_DATASETS for s in ("train", "validation")
               if not (ROOT / "data/processed" / d / f"{s}.jsonl").exists()]
    if missing:
        raise FileNotFoundError("Missing canonical datasets:\n" + "\n".join(missing))
    suffix = part_suffix(args.part)
    error_log = error_log_path(args.part)
    print(f"GPU {args.gpu}: {torch.cuda.get_device_name(0)}; versions: {versions}", flush=True)
    print(f"Checking {len(checked)} engines"
          f"{f' for partition {args.part}' if args.part else ''}: {', '.join(checked)}", flush=True)
    for model in checked:
        smoke(model, args.gpu, error_log)
    report = {"complete": True, "part": args.part, "versions": versions,
              "gpu": torch.cuda.get_device_name(0), "gpu_index": args.gpu,
              "flashinfer": flashinfer_report, "hub_access": access, "skipped_models": skipped,
              "models": {key: MODELS[key].repo_id for key in checked}}
    path = ROOT / f".run_state/validation_preflight{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    error_log.unlink(missing_ok=True)
    print(f"All {len(checked)} vLLM engines passed. Smoke generations are not experiment outcomes.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--part", help="Check only this partition's models, plus the fixed judge")
    parser.add_argument("--model", help="Internal: smoke-check just this model in an isolated process")
    parser.add_argument("--skip-gated", action="store_true",
                        help="Leave out checkpoints this token cannot read instead of stopping")
    args = parser.parse_args()
    if args.model:
        # A child's output is captured and recorded by its parent, not by itself.
        run(args)
        return
    try:
        run(args)
    except PreflightFailure:
        raise  # smoke() already wrote the model's output and its diagnosis
    except BaseException as exc:
        # Everything before the model loop lands here: FlashInfer, a missing
        # package, no CUDA, absent datasets. Those used to reach only the log.
        record_failure(error_log_path(args.part), traceback.format_exc(), stage="preflight",
                       gpu=args.gpu, part=args.part, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
