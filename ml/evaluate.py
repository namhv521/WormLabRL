"""Evaluate an agent on a locked dataset manifest (SPEC-02 §7.2).

    python -m ml.evaluate --model-id <id> --dataset data/manifests/test-v1.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from worm_core import level_from_dict

from .agents import RLAgent, run_episode

TIERS = ("easy", "medium", "hard")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def load_levels(manifest: dict):
    return [(r["difficulty_label"], level_from_dict(r)) for r in manifest["levels"]]


def evaluate(agent, levels) -> dict:
    """Return per-tier + overall metrics; tiers with no levels report N/A (None)."""
    per = defaultdict(lambda: {"episodes": 0, "successes": 0, "steps": [], "fail": Counter(), "lat": []})
    for tier, L in levels:
        r = run_episode(agent, L)
        b = per[tier]
        b["episodes"] += 1
        b["lat"].append(r.get("p95_act_ms", 0.0))
        if r["outcome"] == "SOLVED":
            b["successes"] += 1
            b["steps"].append(r["steps"])
        else:
            b["fail"][r["outcome"]] += 1

    def summarize(b):
        n, k = b["episodes"], b["successes"]
        return {"episodes": n, "successes": k,
                "success_rate": round(k / n, 4) if n else None,
                "wilson95": wilson(k, n),
                "median_steps_solved": statistics.median(b["steps"]) if b["steps"] else None,
                "mean_steps_solved": round(statistics.mean(b["steps"]), 2) if b["steps"] else None,
                "timeouts": b["fail"]["TIMEOUT"], "deadlocks": b["fail"]["DEADLOCK"],
                "p95_inference_latency_ms": round(max(b["lat"]), 3) if b["lat"] else None}

    out = {t: summarize(per[t]) if t in per else None for t in TIERS}
    allb = {"episodes": 0, "successes": 0, "steps": [], "fail": Counter(), "lat": []}
    for b in per.values():
        for k in ("episodes", "successes"):
            allb[k] += b[k]
        allb["steps"] += b["steps"]
        allb["fail"] += b["fail"]
        allb["lat"] += b["lat"]
    out["overall"] = summarize(allb)
    return out


def main(argv=None):
    from .registry import MODELS_DIR, load_model
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--dataset", default="data/manifests/test-v1.json")
    a = p.parse_args(argv)

    from generator.build import load_manifest
    m = load_manifest(a.dataset)
    model, mf = load_model(a.model_id)
    res = evaluate(RLAgent(model, a.model_id), load_levels(m))
    report = {"model_id": a.model_id, "agent": "rl", "dataset": Path(a.dataset).name, "split": m["split"],
              "dataset_hash": m["dataset_hash"], "seed": mf.get("seed"), "run_id": mf.get("run_id"),
              "summary": {t: (v and {"successes": v["successes"], "total": v["episodes"]}) for t, v in res.items()},
              "metrics": res}
    if m["split"] == "test":
        (MODELS_DIR / a.model_id / "evaluation.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
