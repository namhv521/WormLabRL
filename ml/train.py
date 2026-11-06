"""Train MaskablePPO with validation-gated curriculum (SPEC-02 §3, §5.2, §6).

    python -m ml.train --config configs/train/maskable_ppo.yaml --seed 11

Only train + validation manifests are read; the test manifest is never opened.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from worm_core import RULESET_ID, canonical_json, level_from_dict

from .agents import RLAgent
from .evaluate import evaluate
from .registry import ROOT, RUNS_DIR, register, sha256_file

STAGES = [
    {"name": "C0", "mix": {"easy": 1.0}, "gate": ("easy", 0.80)},
    {"name": "C1", "mix": {"easy": 0.6, "medium": 0.4}, "gate": ("medium", 0.60)},
    {"name": "C2", "mix": {"easy": 0.3, "medium": 0.4, "hard": 0.3}, "gate": None},
]
GATE_STREAK = 3


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def stage_weights(tiers, mix):
    """Per-level weights so each tier gets its mix share regardless of tier size."""
    counts = {t: tiers.count(t) for t in set(tiers)}
    return [mix.get(t, 0.0) / counts[t] for t in tiers]


def build(cfg, seed, train_levels, train_tiers):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rl_env import WormEnv

    def make(i):
        def _f():
            e = WormEnv(train_levels, weights=stage_weights(train_tiers, STAGES[0]["mix"]))
            e.reset(seed=seed * 1000 + i)
            return e
        return _f

    venv = DummyVecEnv([make(i) for i in range(cfg["n_envs"])])
    venv.seed(seed)
    model = MaskablePPO("MlpPolicy", venv, seed=seed, device=cfg.get("device", "cpu"), verbose=0,
                        policy_kwargs={"net_arch": cfg["net_arch"]}, **cfg["ppo"])
    return model, venv


def main(argv=None):
    from sb3_contrib.common.maskable.utils import get_action_masks  # noqa: F401  (import check)
    from stable_baselines3.common.callbacks import BaseCallback

    from generator.build import load_manifest

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/train/maskable_ppo.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timesteps", type=int, help="override total_timesteps")
    p.add_argument("--tag", default="")
    a = p.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if a.timesteps:
        cfg["total_timesteps"] = a.timesteps
    config_hash = hashlib.sha256(canonical_json(cfg).encode()).hexdigest()

    train_m = load_manifest(cfg["train_manifest"])
    val_m = load_manifest(cfg["validation_manifest"])
    if {"test"} & {train_m["split"], val_m["split"]}:
        raise SystemExit("refusing to train/validate on the test split")
    train_h = {r["canonical_hash"] for r in train_m["levels"]}
    if train_h & {r["canonical_hash"] for r in val_m["levels"]}:
        raise SystemExit("train/validation canonical-hash leakage")

    train_levels = [level_from_dict(r) for r in train_m["levels"]]
    train_tiers = [r["difficulty_label"] for r in train_m["levels"]]
    val_by_tier = {}
    for r in val_m["levels"]:
        val_by_tier.setdefault(r["difficulty_label"], []).append((r["difficulty_label"], level_from_dict(r)))

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"ppo-s{a.seed}-{stamp}{('-' + a.tag) if a.tag else ''}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    model, venv = build(cfg, a.seed, train_levels, train_tiers)
    metrics_f = open(run_dir / "metrics.jsonl", "w", encoding="utf-8")
    state = {"stage": 0, "streak": 0, "best": -1.0, "gates": []}

    class Curriculum(BaseCallback):
        def __init__(self):
            super().__init__()
            self.ep_rewards, self.ep_lens, self.ep_wins = [], [], []
            self.next_eval = cfg["eval_freq"]

        def _on_step(self):
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if done:
                    ep = info.get("episode")
                    if ep:
                        self.ep_rewards.append(ep["r"])
                        self.ep_lens.append(ep["l"])
                    self.ep_wins.append(bool(info.get("won")))
            if self.num_timesteps >= self.next_eval:
                self.next_eval += cfg["eval_freq"]
                self.evaluate()
            return True

        def evaluate(self):
            st = STAGES[state["stage"]]
            agent = RLAgent(self.model)
            tiers = sorted(st["mix"])
            res = evaluate(agent, [x for t in tiers for x in val_by_tier.get(t, [])])
            rec = {"timesteps": self.num_timesteps, "stage": st["name"],
                   "train_win_rate": round(float(np.mean(self.ep_wins)), 4) if self.ep_wins else None,
                   "train_mean_len": round(float(np.mean(self.ep_lens)), 2) if self.ep_lens else None,
                   "val": {t: res[t]["success_rate"] for t in tiers if res.get(t)}}
            self.ep_rewards, self.ep_lens, self.ep_wins = [], [], []
            overall = res["overall"]["success_rate"] or 0.0
            # Best checkpoint is tracked within the final stage (or the furthest stage reached).
            score = state["stage"] * 10 + overall
            if score > state["best"]:
                state["best"] = score
                self.model.save(run_dir / "checkpoints" / "best.zip")
                rec["saved_best"] = True
            if st["gate"]:
                tier, thr = st["gate"]
                ok = (res.get(tier) or {}).get("success_rate", 0.0) >= thr
                state["streak"] = state["streak"] + 1 if ok else 0
                if state["streak"] >= GATE_STREAK:
                    state["gates"].append({"stage": st["name"], "passed_at": self.num_timesteps})
                    state["stage"] += 1
                    state["streak"] = 0
                    w = stage_weights(train_tiers, STAGES[state["stage"]]["mix"])
                    self.training_env.env_method("set_weights", w)
                    rec["advanced_to"] = STAGES[state["stage"]]["name"]
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            print(json.dumps(rec), flush=True)

    from stable_baselines3.common.vec_env import VecMonitor
    model.set_env(VecMonitor(venv))
    t0 = time.time()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    model.learn(total_timesteps=cfg["total_timesteps"], callback=Curriculum(), progress_bar=False)
    model.save(run_dir / "checkpoints" / "final.zip")
    metrics_f.close()

    import gymnasium
    import sb3_contrib
    import stable_baselines3
    import torch
    best = run_dir / "checkpoints" / "best.zip"
    manifest = {
        "run_id": run_id, "git_commit": git_commit(), "ruleset_id": RULESET_ID, "config": cfg,
        "config_hash": config_hash, "seed": a.seed,
        "dataset_hashes": {"train": train_m["dataset_hash"], "validation": val_m["dataset_hash"]},
        "started_at": started, "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - t0, 1), "train_steps": int(model.num_timesteps),
        "device": str(model.device), "platform": platform.platform(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__,
                     "sb3_contrib": sb3_contrib.__version__, "torch": torch.__version__},
        "curriculum": {"final_stage": STAGES[state["stage"]]["name"], "gates_passed": state["gates"],
                       "gates_not_passed": [s["name"] for s in STAGES[state["stage"]:] if s["gate"]]},
        "checkpoints": {"best": sha256_file(best) if best.exists() else None,
                        "final": sha256_file(run_dir / "checkpoints" / "final.zip")},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    model_id = f"{run_id}-best"
    register(best if best.exists() else run_dir / "checkpoints" / "final.zip", model_id,
             {"run_id": run_id, "seed": a.seed, "config_hash": config_hash, "git_commit": manifest["git_commit"],
              "train_steps": manifest["train_steps"], "trained_at": manifest["ended_at"],
              "dataset_hashes": manifest["dataset_hashes"]})
    print(f"run {run_id} -> model {model_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
