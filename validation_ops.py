#!/usr/bin/env python
"""One-command local validation, optional teacher completion and separate validation handoffs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import experiment_ops as ops


class ValidationNotReady(RuntimeError):
    """Expected absence of a run id, not a failure of the status command."""


def validation_id(explicit=None, incoming=False):
    if explicit:
        return ops.active_id(explicit)
    local = ops.STATE / "validation_experiment_id"
    shared = ops.SHARED / "current_validation.json"
    # During prepare the id is written before any model generation finishes.
    # With partitions there is one job file per GPU; any of them being in a
    # validation-local run is enough to prefer the id that run is writing.
    jobs = [state for _, state in ops.job_states() if state.get("stage") == "validation-local"]
    running_id = ops.STATE / "experiment_id"
    if (not incoming and jobs and not running_id.exists()
            and any(state.get("status") in ("starting", "running", "failed") for state in jobs)):
        raise ValidationNotReady("Validation ID not assigned yet. It is created when prepare starts, after preflight. See .run_state/validation-local*.log")
    if not incoming and jobs and running_id.exists():
        return ops.active_id(running_id.read_text(encoding="utf-8").strip())
    candidates = [shared, local] if incoming else [local, shared]
    for path in candidates:
        if path.exists():
            value = ops.read(path)["experiment_id"] if path.suffix == ".json" else path.read_text(encoding="utf-8").strip()
            return ops.active_id(value)
    raise ValidationNotReady("No validation run selected; start local, pull its handoff, or pass --experiment-id")


def part_receipt(experiment_id, stage, part):
    path = ops.receipt(experiment_id, stage)
    return path.with_name(path.name.replace(".json", f".{part}.json"))


def show_status(explicit=None):
    jobs = [state for _, state in ops.job_states()]
    try:
        experiment_id = validation_id(explicit)
    except ValidationNotReady as exc:
        for job in jobs:
            print(json.dumps(job, indent=2))
        print(str(exc))
        return
    exchange, _ = ops.paths(experiment_id)
    manifest_path = exchange / "manifest.json"
    if manifest_path.exists() and ops.read(manifest_path).get("eval_split") != "validation":
        raise ValueError("The selected run is not validation")
    print(f"Validation: {experiment_id}")
    for job in jobs:
        print(json.dumps(job, indent=2))
    for stage in ops.STAGES:
        path = ops.receipt(experiment_id, stage)
        done = path.exists() and ops.read(path).get("complete")
        # Before merge, a stage can be finished on one GPU and still running on
        # the other; say which, instead of a flat "pending".
        parts = [part for part in ops.PARTS
                 if part_receipt(experiment_id, stage, part).exists()
                 and ops.read(part_receipt(experiment_id, stage, part)).get("complete")]
        detail = "" if done or not parts else f" (partitions done: {', '.join(parts)}; run merge)"
        print(f"{stage}: {'complete' if done else 'pending'}{detail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "share", "restore", "analyze", "merge"))
    parser.add_argument("stage", nargs="?", choices=("local", *ops.STAGES))
    parser.add_argument("--experiment-id")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--pack-only", action="store_true")
    parser.add_argument("--part", choices=ops.PARTS,
                        help="Run only this partition's models on --gpu, so two GPUs share one run.")
    parser.add_argument("--skip-gated", action="store_true",
                        help="Leave out checkpoints this token cannot read yet; the same command fills them in later.")
    for name in ops.PARTS:
        parser.add_argument(f"--{name}", dest="part", action="store_const", const=name,
                            help=f"Shorthand for --part {name}")
    args = parser.parse_args()
    if args.action == "status":
        show_status(args.experiment_id)
        return
    if args.part and args.action != "start":
        parser.error("--part applies to start only; merge and share act on the whole run")
    if args.action == "start" and args.stage == "local":
        if args.experiment_id:
            parser.error("local derives the run id from the frozen preparation parameters; do not pass an id")
        ops.start("validation-local", None, args.gpu, part=args.part, skip_gated=args.skip_gated)
        return
    if args.action == "merge":
        experiment_id = validation_id(args.experiment_id)
        subprocess.run([sys.executable, "-u", "run_experiment.py", "merge",
                        "--experiment-id", experiment_id], cwd=ops.ROOT, check=True)
        subprocess.run([sys.executable, "analyze_validation.py", "--experiment-id", experiment_id],
                       cwd=ops.ROOT, check=True)
        return
    if args.action == "start" and args.stage == "prepare":
        parser.error("Use start local to prepare the validation preset and evaluate self-reflection")
    if args.action not in ("status", "analyze") and args.stage not in ops.STAGES:
        parser.error("Specify prepare, self-eval, teacher or finish")
    incoming = args.action == "restore" or (args.action == "start" and args.stage in ("teacher", "finish"))
    experiment_id = validation_id(args.experiment_id, incoming)
    exchange, results = ops.paths(experiment_id)
    manifest_path = exchange / "manifest.json"
    if args.action == "start" and args.stage in ("teacher", "finish"):
        prior = "prepare" if args.stage == "teacher" else "teacher"
        if (ops.SHARED / experiment_id / prior / "bundle.json").exists():
            ops.restore(experiment_id, prior)
    if manifest_path.exists() and ops.read(manifest_path).get("eval_split") != "validation":
        raise ValueError("The selected run is not validation")
    if args.action == "start":
        manifest = ops.read(manifest_path)
        if manifest.get("teacher_role") != "teacher-only" or manifest.get("experiment_preset") != "validation-threshold":
            raise ValueError("Use the validation-threshold preset: this command never starts GPT reference evaluations")
    if args.action == "analyze":
        command = [sys.executable, "analyze_validation.py", "--experiment-id", experiment_id]
        subprocess.run(command, cwd=ops.ROOT, check=True)
    elif args.action == "restore":
        ops.restore(experiment_id, args.stage)
    elif args.action == "share":
        ops.share(experiment_id, args.stage, publish=not args.pack_only)
    elif args.action == "start":
        ops.start(args.stage, experiment_id, args.gpu,
                  restore_artifacts=args.stage not in ("teacher", "finish"), part=args.part,
                  skip_gated=args.skip_gated)


if __name__ == "__main__":
    main()
