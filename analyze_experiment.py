#!/usr/bin/env python
"""Export filtered outcomes and coverage without altering the experiment."""
import argparse
from pathlib import Path

from rmcq.analysis import filter_outcomes
from run_experiment import load_jsonl, save_jsonl, save_json, save_csv, split_csv, json_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--exclude-flags", default="")
    parser.add_argument("--exclude-methods", default="")
    parser.add_argument("--conditions", default="")
    parser.add_argument("--race-subsets", default="")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--resolved-only", action="store_true")
    args = parser.parse_args()
    config = {name: split_csv(getattr(args, name)) for name in
              ("exclude_flags", "exclude_methods", "conditions", "race_subsets")}
    config.update(paired=args.paired, resolved_only=args.resolved_only)
    rows, audit = filter_outcomes(load_jsonl(args.outcomes), **config)
    folder = args.outcomes.parent / "views" / json_hash(config)
    save_json(folder / "filters.json", config)
    save_jsonl(folder / "outcomes.jsonl", rows)
    save_csv(folder / "coverage.csv", audit)
    print(f"{len(rows)} selected outcomes -> {folder}")


if __name__ == "__main__":
    main()
