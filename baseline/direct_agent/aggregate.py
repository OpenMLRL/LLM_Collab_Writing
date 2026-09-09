"""Require two complete seeds per method; never average pending/partial runs."""
import argparse
import json
from pathlib import Path
import statistics
from .protocol import ALGORITHMS


def aggregate(paths, seeds=(42, 43)):
    runs = {}
    protocols = set()
    for path in map(Path, paths):
        complete = json.loads((path / "completed.json").read_text())
        final = json.loads((path / "eval_final.json").read_text())
        initial = json.loads((path / "eval_initial.json").read_text())
        cfg = json.loads((path / "config.json").read_text())
        key = (complete["algorithm"], complete["seed"])
        if key in runs or (path / "failed.json").exists() or complete["status"] != "finished":
            raise ValueError("Duplicate or failed run")
        if final["phase"] != "final" or final["count"] != cfg["eval_samples"]:
            raise ValueError("Incomplete full evaluation")
        if any(r["algorithm"] != key[0] or r["seed"] != key[1] for r in (initial, final)):
            raise ValueError("Mismatched run identity")
        protocols.add(final["protocol_sha256"])
        runs[key] = (initial, final)
    if set(runs) != {(a, s) for a in ALGORITHMS for s in seeds} or len(protocols) != 1:
        raise ValueError("Expected exactly two matched methods and requested seeds")
    output = {}
    for algorithm in ALGORITHMS:
        output[algorithm] = {}
        for metric in runs[algorithm, seeds[0]][1]["metrics"]:
            values = [runs[algorithm, s][1]["metrics"][metric] for s in seeds]
            deltas = [runs[algorithm, s][1]["metrics"][metric] - runs[algorithm, s][0]["metrics"][metric]
                      for s in seeds]
            output[algorithm][metric] = dict(mean=statistics.mean(values),
                sample_std=statistics.stdev(values), mean_delta_from_initial=statistics.mean(deltas), n=len(values))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.paths), indent=2))
