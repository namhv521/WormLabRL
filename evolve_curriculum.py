"""
Closed-Loop Co-Evolution Dojo CLI:
PCGRL Designer generates progressively harder solvable maps ->
RL Solver attempts Zero-Shot -> RL Solver self-trains right after each map ->
Difficulty automatically escalates (Tier 1 -> Tier 2 -> Tier 3 -> Tier 4 -> Tier 5)!

Example:
  python evolve_curriculum.py --rounds 8 --out generated_maps/coevolution_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from train_product import DEFAULT_CKPT, DEFAULT_DESIGNER_CKPT
from worm_physics import WormMap
from worm_ai.model import WormBrainNet, load_brain_checkpoint, save_brain_checkpoint
from worm_ai.pcgrl import DesignerPolicyNet, ProgressivePCGRLGenerator, load_designer_checkpoint
from worm_ai.physics import LevelArena, ROOT
from worm_ai.trainer import ContinualWormTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Co-Evolutionary Self-Play & Progressive Map Generation")
    parser.add_argument("--rounds", type=int, default=8, help="Number of progressive co-evolution stages")
    parser.add_argument("--solver-ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--designer-ckpt", type=Path, default=DEFAULT_DESIGNER_CKPT)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "generated_maps" / "coevolution_run")
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.from_scratch and args.solver_ckpt.exists():
        solver_net, _ = load_brain_checkpoint(args.solver_ckpt)
        print(f"[Init] Loaded Solver Brain from {args.solver_ckpt}")
    else:
        solver_net = WormBrainNet()
        print("[Init] Starting Solver Brain from scratch")

    if args.designer_ckpt.exists():
        designer_net, _ = load_designer_checkpoint(args.designer_ckpt)
        print(f"[Init] Loaded PCGRL Designer from {args.designer_ckpt}")
    else:
        designer_net = DesignerPolicyNet()

    trainer = ContinualWormTrainer(model=solver_net, seed=args.seed)
    generator = ProgressivePCGRLGenerator(
        model=designer_net,
        archive_path=args.out / "archive.json",
        seed=args.seed,
    )

    def solver_zero_shot_check(lv_dict: dict) -> bool:
        arena = LevelArena(WormMap(lv_dict), name="probe", level_dict=lv_dict)
        return trainer.play_episode(arena, greedy=True, max_steps=180, record_frames=False).won

    current_tier = 1
    difficulty_floor = 12.0
    consecutive_masteries = 0
    logs = []
    generated_pack = []
    t0 = time.perf_counter()

    print(f"\n=== Starting Co-Evolution Dojo ({args.rounds} Escalating Stages) ===")
    for r in range(1, args.rounds + 1):
        lv, ores = generator.generate_verified_map(
            target_tier=current_tier,
            min_difficulty_floor=difficulty_floor,
            max_attempts=50,
            solver_eval_fn=solver_zero_shot_check,
        )
        arena = LevelArena(WormMap(lv), name=lv["name"], level_dict=lv)
        summary = trainer.play_and_self_train_stage(
            arena, ppo_rounds=4, envs=8, horizon=40, record_frames=False
        )

        # Escalate difficulty floor & tier whenever Solver conquers the generated stage!
        prev_tier = current_tier
        if summary["after_train_won"]:
            consecutive_masteries += 1
            difficulty_floor = max(difficulty_floor + 4.5, ores.difficulty_score + 1.0)
            if consecutive_masteries >= 2 and current_tier < 5:
                current_tier += 1
                consecutive_masteries = 0

        record = {
            "round": r,
            "map_name": lv["name"],
            "requested_tier": prev_tier,
            "actual_tier": ores.difficulty_tier,
            "tier_name": ores.tier_name,
            "difficulty_score": ores.difficulty_score,
            "optimal_steps": ores.optimal_steps,
            "mandatory_apples": ores.mandatory_apples,
            "mandatory_stones": ores.mandatory_stones,
            "is_deceptive": ores.is_deceptive,
            "solver_zero_shot_won": summary["zero_shot_won"],
            "solver_after_train_won": summary["after_train_won"],
            "solver_steps": summary["after_train_steps"],
            "next_tier": current_tier,
            "next_difficulty_floor": round(difficulty_floor, 1),
        }
        logs.append(record)
        generated_pack.append(dict(lv, oracle=ores.to_dict()))
        print(
            f"[Round {r:02d}/{args.rounds}] Map {lv['name']:<10} (Tier {ores.difficulty_tier} {ores.tier_name:<11} "
            f"Score={ores.difficulty_score:>4.1f}, L*={ores.optimal_steps:>2d}) | "
            f"Zero-Shot={'WIN ' if summary['zero_shot_won'] else 'FAIL'} -> "
            f"Self-Trained={'WIN ' if summary['after_train_won'] else 'FAIL'} ({summary['after_train_steps']:>2d} steps) | "
            f"Next Tier -> Lv.{current_tier} (floor>={difficulty_floor:.1f})"
        )

    evolved_ckpt = args.out / "worm_brain_coevolved.pt"
    save_brain_checkpoint(evolved_ckpt, trainer.model, trainer.optimizer, metadata={"coevolution": logs})
    (args.out / "coevolution_metrics.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    (args.out / "coevolved_maps.json").write_text(
        json.dumps({"levelsA": generated_pack, "levelsB": []}, indent=2), encoding="utf-8"
    )
    print(
        f"\n=== Co-Evolution Complete in {time.perf_counter() - t0:.2f}s ===\n"
        f"  Final Difficulty Reached : Tier {current_tier} (floor {difficulty_floor:.1f})\n"
        f"  Post-Stage Mastery Rate  : {sum(x['solver_after_train_won'] for x in logs)}/{len(logs)}\n"
        f"  Saved Evolved Brain      -> {evolved_ckpt}"
    )


if __name__ == "__main__":
    main()
