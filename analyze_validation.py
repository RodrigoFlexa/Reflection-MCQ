#!/usr/bin/env python
"""Generate validation threshold tables and figures offline, including partial self-eval."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--fallback", action="store_true", help="Use baseline on failures/below threshold (default: exclude)")
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--min-coverage", type=float, default=.1)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from rmcq.test_analysis import load_run
    from rmcq.validation_analysis import study, export_study, plot_study, CLEAN_FLAGS, THRESHOLDS
    root = Path(__file__).resolve().parent
    frame, sources = load_run(root, args.experiment_id, "validation")
    if frame.empty:
        raise FileNotFoundError("No validation outcomes; finish self-eval first, or restore its handoff")
    config = dict(experiment_id=args.experiment_id, fallback=args.fallback, min_n=args.min_n,
                  min_coverage=args.min_coverage, clean_flags=CLEAN_FLAGS, thresholds=THRESHOLDS, sources=sources)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    directory = root / "data/results/reflection_top1" / args.experiment_id / "analysis" / f"validation_{digest}"
    print(f"Analyzing {len(frame)} existing validation outcomes -> {directory}", flush=True)
    views = study(frame, fallback=args.fallback, min_n=args.min_n, min_coverage=args.min_coverage)
    export_study(views, directory)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    plot_study(views, directory, show=False)
    print(f"Validation plots complete: {directory}", flush=True)


if __name__ == "__main__":
    main()
