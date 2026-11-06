"""Random / Greedy / BFS baselines on a dataset manifest (SPEC-02 §7.1).

    python -m ml.baselines --dataset data/manifests/test-v1.json --agents random,greedy,bfs
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .agents import BFSAgent, GreedyAgent, RandomAgent
from .evaluate import TIERS, evaluate, load_levels
from .registry import ROOT

RANDOM_EVAL_SEEDS = (0, 1, 2)
OUT = ROOT / "artifacts" / "baselines"


def run(names, manifest) -> dict:
    levels = load_levels(manifest)
    out = {}
    for name in names:
        if name == "random":
            runs = [evaluate(RandomAgent(s), levels) for s in RANDOM_EVAL_SEEDS]
            out["random"] = {"eval_seeds": list(RANDOM_EVAL_SEEDS), "runs": runs,
                             "mean_success_rate": {t: (round(statistics.mean(r[t]["success_rate"] for r in runs), 4)
                                                       if runs[0][t] else None) for t in (*TIERS, "overall")}}
        elif name == "greedy":
            out["greedy"] = evaluate(GreedyAgent(), levels)
        elif name == "bfs":
            out["bfs"] = evaluate(BFSAgent(), levels)
        else:
            raise SystemExit(f"unknown agent {name!r}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/manifests/test-v1.json")
    p.add_argument("--agents", default="random,greedy,bfs")
    a = p.parse_args(argv)
    from generator.build import load_manifest
    m = load_manifest(a.dataset)
    res = run(a.agents.split(","), m)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"dataset": Path(a.dataset).name, "split": m["split"], "dataset_hash": m["dataset_hash"], "agents": res}
    path = OUT / f"{m['split']}-{m['dataset_hash'][:12]}.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for name, r in res.items():
        rates = r["mean_success_rate"] if name == "random" else {t: r[t] and r[t]["success_rate"] for t in (*TIERS, "overall")}
        print(f"{name:<7} {rates}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
