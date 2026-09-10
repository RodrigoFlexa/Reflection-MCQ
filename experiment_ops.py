#!/usr/bin/env python
"""Start, inspect and share the GPU -> Petrobras -> GPU experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".run_state"
SHARED = ROOT / "experiment_handoff"
STAGES = ("prepare", "self-eval", "teacher", "finish")
JOB_STAGES = STAGES + ("validation-local", "merge")
PARTS = ("p1", "p2")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def suffix(part):
    return "" if part is None else f".{part}"


def job_path(part=None):
    """One job file and one lock per partition, so two GPUs share a checkout."""
    return STATE / f"job{suffix(part)}.json"


def job_states():
    """Every job file that exists now, as (part, state) pairs."""
    return [(part, read(job_path(part))) for part in (None, *PARTS) if job_path(part).exists()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def git(*args, capture=False):
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def active_id(explicit=None, shared_first=False):
    if explicit:
        value = explicit
    elif shared_first and (SHARED / "current.json").exists():
        value = read(SHARED / "current.json")["experiment_id"]
    elif (STATE / "experiment_id").exists():
        value = (STATE / "experiment_id").read_text(encoding="utf-8").strip()
    elif (SHARED / "current.json").exists():
        value = read(SHARED / "current.json")["experiment_id"]
    else:
        raise RuntimeError("No experiment id yet. Start prepare, or pull its published handoff first.")
    if not re.fullmatch(r"[0-9a-f]{12}", value):
        raise ValueError("Invalid experiment id")
    return value


def paths(experiment_id):
    return ROOT / "experiment_exchange" / experiment_id, ROOT / "data/results/reflection_top1" / experiment_id


def receipt(experiment_id, stage):
    exchange, results = paths(experiment_id)
    if stage == "self-eval":
        return results / "self_eval" / "self_eval_receipt.json"
    return (results if stage == "finish" else exchange) / f"{stage}_receipt.json"


def require_complete(experiment_id, stage):
    path = receipt(experiment_id, stage)
    if not path.exists() or read(path).get("complete") is not True:
        raise RuntimeError(f"{stage} is not complete for {experiment_id}. See experiment_ops.py status.")


def restore(experiment_id, stage):
    from rmcq.handoff import unpack
    result = unpack(ROOT, SHARED / experiment_id / stage, experiment_id, stage)
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "experiment_id").write_text(experiment_id + "\n", encoding="utf-8")
    print(f"Restored {stage}: {experiment_id}, {len(result['files'])} verified files", flush=True)


def share(experiment_id, stage, publish=True):
    from rmcq.handoff import pack
    for part, latest in job_states():
        if latest.get("stage") in (stage, "validation-local") and latest.get("status") != "complete":
            label = "The latest local job" if part is None else f"Partition {part}"
            raise RuntimeError(f"{label} for this stage has not completed; inspect its log before sharing.")
    require_complete(experiment_id, stage)
    exchange, results = paths(experiment_id)
    if stage == "prepare":
        files = [p for p in exchange.rglob("*") if p.is_file() and "teacher" not in p.relative_to(exchange).parts
                 and p.name != "teacher_receipt.json" and p.suffix in (".json", ".jsonl")]
        race = ROOT / "data/processed/race"
        files += [race / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "dataset_manifest.json")]
    elif stage == "teacher":
        files = [p for p in (exchange / "teacher").rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl")]
        files += [exchange / "teacher_receipt.json", exchange / "manifest.json"]
    elif stage == "self-eval":
        files = [p for p in (results / "self_eval").rglob("*")
                 if p.is_file() and p.suffix in (".json", ".jsonl", ".csv")]
        files += [exchange / "manifest.json"]
    else:
        files = [p for folder in (results / "analysis", results / "models") for p in folder.rglob("*")
                 if p.is_file() and p.suffix in (".json", ".jsonl", ".csv")]
        files += [results / "finish_receipt.json"]
    if publish and git("diff", "--cached", "--name-only", capture=True):
        raise RuntimeError("There are already staged Git changes. Commit or unstage them before sharing experiment data.")
    destination = SHARED / experiment_id / stage
    # Audit the executable source, independently of subsequent data-only commits.
    revision = git("rev-parse", "HEAD", capture=True)
    manifest = pack(ROOT, files, destination, {"experiment_id": experiment_id, "stage": stage, "git_commit": revision})
    write(SHARED / "current.json", {"experiment_id": experiment_id, "stage": stage, "git_commit": revision})
    validation = read(exchange / "manifest.json").get("eval_split") == "validation"
    if validation:
        write(SHARED / "current_validation.json", {"experiment_id": experiment_id, "stage": stage, "git_commit": revision})
    print(f"Packed {len(files)} files into {len(manifest['parts'])} parts, at most 20 MiB each.", flush=True)
    if publish:
        # Stage only the handoff; never credentials, caches or unrelated changes.
        git("add", "--", destination.relative_to(ROOT).as_posix(), "experiment_handoff/current.json",
            *(["experiment_handoff/current_validation.json"] if validation else []))
        if git("diff", "--cached", "--name-only", capture=True):
            git("commit", "-m", f"data: share {stage} for experiment {experiment_id}")
        git("push", "origin", "HEAD")
        print("GitHub handoff published.")


def is_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def frozen_environment(manifest):
    environment = os.environ.copy()
    mapping = {
        "max_model_len": "RMCQ_MAX_MODEL_LEN", "generation_seed": "RMCQ_SEED",
        "dtype": "RMCQ_DTYPE", "vllm_deterministic": "RMCQ_VLLM_DETERMINISTIC",
        "vllm_max_num_seqs": "RMCQ_VLLM_MAX_NUM_SEQS", "azure_max_tokens": "RMCQ_AZURE_MAX_TOKENS",
        "azure_reasoning_min_tokens": "RMCQ_AZURE_REASONING_MIN_TOKENS",
        "azure_reasoning_effort": "RMCQ_AZURE_REASONING_EFFORT",
    }
    for key, variable in mapping.items():
        value = manifest["runtime_limits"][key]
        environment[variable] = "none" if value is None else str(int(value)) if isinstance(value, bool) else str(value)
    return environment


def work(stage, experiment_id, gpu, lock_fd=None, part=None):
    job = job_path(part)
    state = read(job)
    state.update(pid=os.getpid(), status="running", started_at=time.time())
    write(job, state)
    partition = ["--part", part] if part else []
    try:
        if stage == "validation-local":
            # Separate processes release every CUDA engine between phases.
            subprocess.run([sys.executable, "validation_preflight.py", "--gpu", gpu, *partition],
                           cwd=ROOT, check=True)
            subprocess.run([sys.executable, "-u", "run_experiment.py", "prepare",
                            "--preset", "validation-threshold", "--gpu", gpu, *partition,
                            "--write-id", str(STATE / "experiment_id")], cwd=ROOT, check=True)
            experiment_id = active_id()
            (STATE / "validation_experiment_id").write_text(experiment_id + "\n", encoding="utf-8")
            manifest = read(paths(experiment_id)[0] / "manifest.json")
            subprocess.run([sys.executable, "-u", "run_experiment.py", "self-eval",
                            "--experiment-id", experiment_id, "--gpu", gpu, *partition], cwd=ROOT,
                           env=frozen_environment(manifest), check=True)
            # A partition holds only half the models, so plotting it alone would
            # show half a run. `merge` runs the analysis once both are in.
            if part is None:
                subprocess.run([sys.executable, "analyze_validation.py", "--experiment-id", experiment_id],
                               cwd=ROOT, check=True)
            state.update(status="complete", experiment_id=experiment_id, exit_code=0, ended_at=time.time())
            write(job, state)
            if part is not None:
                print(f"Partition {part} complete. When the other partition finishes: "
                      f"python validation_ops.py merge", flush=True)
            return 0
        command = [sys.executable, "-u", "run_experiment.py", stage, "--gpu", gpu, *partition]
        environment = os.environ.copy()
        if stage == "prepare":
            command += ["--eval-split", "test", "--generation-profile", "final", "--backend", "vllm",
                        "--write-id", str(STATE / "experiment_id")]
        else:
            manifest = read(paths(experiment_id)[0] / "manifest.json")
            environment = frozen_environment(manifest)
            command += ["--experiment-id", experiment_id]
        exit_code = subprocess.run(command, cwd=ROOT, env=environment).returncode
        if (exit_code == 0 and part is None and stage in ("finish", "self-eval")
                and manifest.get("experiment_preset") == "validation-threshold"):
            exit_code = subprocess.run([sys.executable, "analyze_validation.py", "--experiment-id", experiment_id],
                                       cwd=ROOT).returncode
        state.update(status="complete" if exit_code == 0 else "failed", exit_code=exit_code, ended_at=time.time())
        try:
            state["experiment_id"] = active_id(experiment_id)
        except RuntimeError:
            pass
        write(job, state)
        return exit_code
    except BaseException as exc:
        state.update(status="failed", error=str(exc), ended_at=time.time())
        write(job, state)
        raise
    finally:
        if lock_fd is not None:
            os.close(lock_fd)


def sibling_running(part):
    """True while another partition's worker is still alive in this checkout."""
    for other, state in job_states():
        if other != part and state.get("status") in ("starting", "running") and is_alive(state.get("pid")):
            return True
    return False


def start(stage, explicit, gpu, restore_artifacts=True, part=None):
    if os.name != "posix":
        raise RuntimeError("Start runs on the Linux GPU/Petrobras server, in its activated Python environment.")
    import fcntl
    STATE.mkdir(parents=True, exist_ok=True)
    # The lock is per partition: p1 and p2 are meant to run side by side here,
    # while a second p1 in the same checkout is still refused.
    fd = os.open(STATE / f"job{suffix(part)}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        label = "A job" if part is None else f"Partition {part}"
        raise RuntimeError(f"{label} is already running in this repository; inspect experiment_ops.py status.") from None
    experiment_id = None
    if stage not in ("prepare", "validation-local"):
        experiment_id = active_id(explicit, shared_first=True)
        previous_stage = "prepare" if stage in ("teacher", "self-eval") else "teacher"
        # Local self-eval can start without a handoff. Other transitions import a newer package if present.
        if restore_artifacts and (SHARED / experiment_id / previous_stage / "bundle.json").exists():
            restore(experiment_id, previous_stage)
        # A partition is gated on its own prepare receipt inside run_experiment,
        # so neither GPU waits at the prepare boundary. The teacher stage is
        # never partitioned, so `finish` still requires the whole teacher.
        if part is None or previous_stage != "prepare":
            require_complete(experiment_id, previous_stage)
    log = STATE / f"{stage}{suffix(part)}.log"
    # Clearing the pointer would erase the id the sibling partition just wrote.
    if stage in ("prepare", "validation-local") and not sibling_running(part):
        (STATE / "experiment_id").unlink(missing_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "work", stage, "--gpu", gpu, "--lock-fd", str(fd)]
    if experiment_id:
        command += ["--experiment-id", experiment_id]
    if part:
        command += ["--part", part]
    # Inherit the OS lock. It is released even if the worker crashes; no PID race.
    write(job_path(part), {"stage": stage, "part": part, "gpu": gpu, "status": "starting",
                           "pid": os.getpid(), "log": str(log)})
    try:
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nStarting {stage}{suffix(part)} on GPU {gpu} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            stream.flush()
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,))
    finally:
        os.close(fd)
    print(f"Started {stage}{suffix(part)} on GPU {gpu}, PID {child.pid}. Disconnecting SSH will not stop it.")
    print(f"Log: {log}\nCheck: python experiment_ops.py status")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "work", "share", "restore", "status"))
    parser.add_argument("stage", choices=JOB_STAGES, nargs="?")
    parser.add_argument("--experiment-id")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--pack-only", action="store_true", help="Prepare a handoff without committing or pushing.")
    parser.add_argument("--part", choices=PARTS, help="Run only this partition of the model grid.")
    for name in PARTS:
        parser.add_argument(f"--{name}", dest="part", action="store_const", const=name,
                            help=f"Shorthand for --part {name}")
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == "status":
        for part, state in job_states():
            print(json.dumps(state, indent=2))
            if os.name == "posix" and state.get("status") == "running" and not is_alive(state.get("pid")):
                label = "Worker" if part is None else f"Partition {part} worker"
                print(f"{label} is no longer running; inspect the log and resume the same stage.")
        try:
            experiment_id = active_id(args.experiment_id)
            print(f"Experiment: {experiment_id}")
            for stage in STAGES:
                path = receipt(experiment_id, stage)
                print(f"{stage}: {'complete' if path.exists() and read(path).get('complete') else 'pending'}")
        except RuntimeError as exc:
            print(exc)
        return
    if args.stage is None:
        parser.error("A stage is required")
    if args.stage in ("validation-local", "merge") and args.action not in ("start", "work"):
        parser.error("Share/restore the prepare or self-eval artifacts of validation-local")
    if args.part and args.stage in ("teacher", "merge"):
        parser.error(f"--part does not apply to {args.stage}; it splits GPU generation only")
    if args.action == "start":
        start(args.stage, args.experiment_id, args.gpu, part=args.part)
    elif args.action == "work":
        sys.exit(work(args.stage, args.experiment_id, args.gpu, args.lock_fd, part=args.part))
    else:
        experiment_id = active_id(args.experiment_id, shared_first=args.action == "restore")
        if args.action == "restore":
            restore(experiment_id, args.stage)
        else:
            share(experiment_id, args.stage, publish=not args.pack_only)


if __name__ == "__main__":
    main()
