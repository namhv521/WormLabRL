"""
Contra-PPO Style Evaluation & Autonomous New-Map Adaptation CLI.

Examples:
  # 1) Test zero-shot + online self-adaptation on the 40 unseen test maps:
  python test_product.py --split test --adapt

  # 2) Test on a specific stage and export HTML replay:
  python test_product.py --world A --stage 22 --adapt --replay-out checkpoints/test_A22.html

  # 3) Test on newly generated PCGRL maps:
  python test_product.py --map-file generated_maps/escalating_pack.json --adapt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from train_product import DEFAULT_CKPT, export_stage_replay_html
from worm_physics import WormMap
from worm_ai.model import WormBrainNet, load_brain_checkpoint
from worm_ai.physics import DEFAULT_LEVELS_JSON, LevelArena, load_dataset, load_levels_json
from worm_ai.trainer import ContinualWormTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WormBrainNet on known or brand-new maps")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--world", type=str.upper, choices=("A", "B"))
    parser.add_argument("--stage", type=int)
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--map-file", type=Path, help="Custom or generated JSON map file to play")
    parser.add_argument(
        "--adapt",
        action="store_true",
        help="Allow autonomous online self-training when encountering a tough unseen map",
    )
    parser.add_argument("--replay-out", type=Path, help="Optional HTML replay path (for single-stage test)")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    if args.checkpoint.exists():
        model, _ = load_brain_checkpoint(args.checkpoint)
        print(f"[Loaded] {args.checkpoint}")
    else:
        model = WormBrainNet()
        print("[Warning] Checkpoint not found; testing with freshly initialized WormBrainNet")

    trainer = ContinualWormTrainer(model=model)

    # Case 1: Custom / Generated JSON file
    if args.map_file:
        pairs = load_levels_json(args.map_file)
        arenas = [
            LevelArena(WormMap(lv), name=lv.get("name", f"{p}{lv['id']}"), level_dict=lv)
            for p, lv in pairs
        ]
    elif args.world and args.stage is not None:
        splits = load_dataset(DEFAULT_LEVELS_JSON)
        target = f"{args.world}{args.stage}"
        all_rows = splits["train"] + splits["val"] + splits["test"]
        match = next((r for r in all_rows if r[0]["name"] == target), None)
        if match is None:
            raise ValueError(f"Stage {target} not found")
        arenas = [match[1]]
    else:
        splits = load_dataset(DEFAULT_LEVELS_JSON)
        arenas = [r[1] for r in splits[args.split]]

    t0 = time.perf_counter()
    zs_wins = 0
    final_wins = 0
    for idx, arena in enumerate(arenas, 1):
        ep = trainer.play_episode(arena, greedy=True, record_frames=bool(args.replay_out))
        zs_wins += int(ep.won)
        if ep.won or not args.adapt:
            final_wins += int(ep.won)
            print(f"[{idx:02d}/{len(arenas)}] {arena.name:<12} | Zero-Shot: {ep.outcome:<5} ({ep.steps} steps)")
            if args.replay_out and len(arenas) == 1:
                export_stage_replay_html(
                    arena,
                    {
                        "zero_shot_outcome": ep.outcome,
                        "zero_shot_steps": ep.steps,
                        "after_train_won": ep.won,
                        "after_train_outcome": ep.outcome,
                        "after_train_steps": ep.steps,
                        "before_frames": ep.frames,
                        "after_frames": ep.frames,
                    },
                    args.replay_out,
                )
        else:
            summary = trainer.play_and_self_train_stage(
                arena, ppo_rounds=3, envs=8, horizon=36, record_frames=bool(args.replay_out)
            )
            final_wins += int(summary["after_train_won"])
            print(
                f"[{idx:02d}/{len(arenas)}] {arena.name:<12} | Zero-Shot: {summary['zero_shot_outcome']:<5} -> "
                f"Self-Adapted: {summary['after_train_outcome']:<5} ({summary['after_train_steps']} steps, {summary['elapsed_sec']}s)"
            )
            if args.replay_out and len(arenas) == 1:
                export_stage_replay_html(arena, summary, args.replay_out)

    dt = time.perf_counter() - t0
    print(
        f"\n=== Evaluation Summary ({len(arenas)} maps, {dt:.2f}s) ===\n"
        f"  Zero-Shot Greedy Wins : {zs_wins}/{len(arenas)} ({zs_wins / max(1, len(arenas)):.1%})\n"
        + (
            f"  With Self-Adaptation  : {final_wins}/{len(arenas)} ({final_wins / max(1, len(arenas)):.1%})\n"
            if args.adapt
            else ""
        )
    )


if __name__ == "__main__":
    main()
