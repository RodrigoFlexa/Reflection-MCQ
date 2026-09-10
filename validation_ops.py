#!/usr/bin/env python
"""One-command local validation, optional teacher completion and separate validation handoffs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import experiment_ops as ops


def validation_id(explicit=None, incoming=False):
    if explicit:
        return ops.active_id(explicit)
    local = ops.STATE / "validation_experiment_id"
    shared = ops.SHARED / "current_validation.json"
    # During prepare the id is written before any model generation finishes.
    job = ops.read(ops.STATE / "job.json") if (ops.STATE / "job.json").exists() else {}
    running_id = ops.STATE / "experiment_id"
    if (not incoming and job.get("stage") == "validation-local"
            and job.get("status") in ("starting", "running") and not running_id.exists()):
        raise RuntimeError("Validation preflight is running; the id is assigned when prepare starts. See .run_state/validation-local.log")
    if not incoming and job.get("stage") == "validation-local" and running_id.exists():
        return ops.active_id(running_id.read_text(encoding="utf-8").strip())
    candidates = [shared, local] if incoming else [local, shared]
    for path in candidates:
        if path.exists():
            value = ops.read(path)["experiment_id"] if path.suffix == ".json" else path.read_text(encoding="utf-8").strip()
            return ops.active_id(value)
    raise RuntimeError("No validation run selected; start local, pull its handoff, or pass --experiment-id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "share", "restore", "analyze"))
    parser.add_argument("stage", nargs="?", choices=("local", *ops.STAGES))
    parser.add_argument("--experiment-id")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--pack-only", action="store_true")
    args = parser.parse_args()
    if args.action == "start" and args.stage == "local":
        if args.experiment_id:
            parser.error("local derives the run id from the frozen preparation parameters; do not pass an id")
        ops.start("validation-local", None, args.gpu)
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
    if args.action == "status":
        print(f"Validation: {experiment_id}")
        if (ops.STATE / "job.json").exists():
            print(json.dumps(ops.read(ops.STATE / "job.json"), indent=2))
        for stage in ops.STAGES:
            path = ops.receipt(experiment_id, stage)
            print(f"{stage}: {'complete' if path.exists() and ops.read(path).get('complete') else 'pending'}")
    elif args.action == "analyze":
        command = [sys.executable, "analyze_validation.py", "--experiment-id", experiment_id]
        subprocess.run(command, cwd=ops.ROOT, check=True)
    elif args.action == "restore":
        ops.restore(experiment_id, args.stage)
    elif args.action == "share":
        ops.share(experiment_id, args.stage, publish=not args.pack_only)
    elif args.action == "start":
        ops.start(args.stage, experiment_id, args.gpu,
                  restore_artifacts=args.stage not in ("teacher", "finish"))


if __name__ == "__main__":
    main()
