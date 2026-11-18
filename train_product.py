"""
Contra-PPO Style Training CLI for Hungry Worm RL Product.

Examples:
  # 1) Train shared foundation brain + PCGRL map designer:
  python train_product.py --mode foundation --out checkpoints

  # 2) Self-play & self-train on a specific stage (like Contra-PPO train.py --world A --stage 67):
  python train_product.py --mode stage --world A --stage 67

  # 3) Sequential Campaign: play stage-by-stage and self-train right after every stage!
  python train_product.py --mode campaign --limit-stages 25 --from-scratch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from build_worm_training_replay import build as build_html_replay
from worm_ai.model import WormBrainNet, load_brain_checkpoint, save_brain_checkpoint
from worm_ai.pcgrl import ProgressivePCGRLGenerator, save_designer_checkpoint
from worm_ai.physics import DEFAULT_LEVELS_JSON, DEFAULT_VIEWER_HTML, ROOT, load_dataset
from worm_ai.trainer import ContinualWormTrainer

DEFAULT_CKPT = ROOT / "checkpoints" / "worm_brain_best.pt"
DEFAULT_DESIGNER_CKPT = ROOT / "checkpoints" / "pcgrl_designer.pt"


def export_stage_replay_html(arena, summary: dict, html_out: Path) -> None:
    """Export before/after stage self-training episodes to an offline interactive HTML viewer."""
    level = arena.to_level_dict()
    entry = {
        "name": arena.name,
        "width": level["width"],
        "height": level["height"],
        "bricks": [b for b in level["bricks"] if b["id"] not in {2, 4, 5, 6}],
        "split": "train",
        "hint_steps": len(level.get("hints", [])),
        "verified_steps": summary["after_train_steps"],
        "verification": "win" if summary["after_train_won"] else summary["after_train_outcome"],
        "attempts": [
            {
                "mode": "before_self_train",
                "outcome": summary["zero_shot_outcome"],
                "steps": summary["zero_shot_steps"],
                "frames": summary["before_frames"],
            },
            {
                "mode": "after_self_train",
                "outcome": summary["after_train_outcome"],
                "steps": summary["after_train_steps"],
                "frames": summary["after_frames"],
            },
        ],
    }
    replay_json = html_out.with_suffix(".replay.json")
    payload = {
        "info": {"stage_self_play": True, "splits": {"train": 1, "test": 0}},
        "maps": [entry],
    }
    replay_json.parent.mkdir(parents=True, exist_ok=True)
    replay_json.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    if DEFAULT_VIEWER_HTML.exists():
        build_html_replay(DEFAULT_VIEWER_HTML, replay_json, html_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hungry Worm Continual RL Trainer (Contra-PPO Style)")
    parser.add_argument("--mode", choices=("foundation", "stage", "campaign"), default="campaign")
    parser.add_argument("--world", type=str.upper, default="A", choices=("A", "B"))
    parser.add_argument("--stage", type=int, default=4, help="Stage number when --mode stage")
    parser.add_argument("--limit-stages", type=int, default=20, help="Number of stages for campaign mode")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--envs", type=int, default=8, help="Parallel environments per stage (Contra-PPO style)")
    parser.add_argument("--ppo-rounds", type=int, default=4, help="PPO self-play rounds per stage")
    parser.add_argument("--warm-epochs", type=int, default=12)
    parser.add_argument("--from-scratch", action="store_true", help="Initialize brand-new untrained brain")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    splits = load_dataset(DEFAULT_LEVELS_JSON)

    if not args.from_scratch and args.checkpoint.exists() and args.mode != "foundation":
        model, meta = load_brain_checkpoint(args.checkpoint)
        print(f"[Init] Loaded existing Shared Brain from {args.checkpoint} (meta={list(meta.keys())})")
    else:
        model = WormBrainNet()
        print(f"[Init] Created fresh WormBrainNet ({model.parameter_count():,} params)")

    trainer = ContinualWormTrainer(model=model, lr=args.lr, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "foundation":
        print("=== [1/2] Training Shared Foundation RL Solver Brain ===")
        hist = trainer.pretrain_foundation(
            splits["train"],
            splits["val"],
            warm_epochs=args.warm_epochs,
            dagger_rounds=2,
            ppo_rounds=2,
        )
        test_zero_shot = trainer.evaluate_split(splits["test"])
        print(
            f"[Benchmark] Held-out Test Split (40 unseen maps) Zero-Shot Greedy: "
            f"{test_zero_shot['wins']}/{test_zero_shot['maps']} ({test_zero_shot['win_rate']:.1%})"
        )
        ckpt_path = args.out / "worm_brain_best.pt"
        save_brain_checkpoint(
            ckpt_path,
            trainer.model,
            trainer.optimizer,
            metadata={"foundation_history": hist, "test_zero_shot": test_zero_shot},
        )
        print(f"[Saved] Solver Brain -> {ckpt_path}")

        print("\n=== [2/2] Training PCGRL Progressive Map Designer ===")
        pcgrl = ProgressivePCGRLGenerator(
            archive_path=ROOT / "generated_maps" / "archive.json",
            seed=args.seed,
        )
        pcg_stats = pcgrl.train_designer_ppo(iterations=18, episodes_per_iter=6, max_edits=16)
        designer_path = args.out / "pcgrl_designer.pt"
        save_designer_checkpoint(designer_path, pcgrl.model, metadata=pcg_stats)
        print(f"[Saved] PCGRL Designer -> {designer_path} (Archive: {pcg_stats['archive_size']} verified maps)")
        return

    if args.mode == "stage":
        target = f"{args.world}{args.stage}"
        all_rows = splits["train"] + splits["val"] + splits["test"]
        match = next((r for r in all_rows if r[0]["name"] == target), None)
        if match is None:
            raise ValueError(f"Stage {target} not found in verified levels")
        _, arena, _ = match
        print(f"=== Self-Playing & Self-Training Stage {target} ({args.envs} parallel envs) ===")
        summary = trainer.play_and_self_train_stage(
            arena,
            ppo_rounds=args.ppo_rounds,
            envs=args.envs,
            horizon=48,
            on_progress=lambda ev: print(
                f"  [{ev['phase']}] outcome={ev.get('greedy_outcome', ev.get('outcome'))} "
                f"steps={ev.get('greedy_steps', ev.get('steps'))}"
            ),
        )
        out_ckpt = args.out / f"worm_brain_{target}.pt"
        save_brain_checkpoint(out_ckpt, trainer.model, trainer.optimizer, metadata=summary)
        replay_html = args.out / f"replay_{target}.html"
        export_stage_replay_html(arena, summary, replay_html)
        print(
            f"[Done] {target}: Before={summary['zero_shot_outcome']}({summary['zero_shot_steps']}) -> "
            f"After={summary['after_train_outcome']}({summary['after_train_steps']}) in {summary['elapsed_sec']}s\n"
            f"       Checkpoint: {out_ckpt}\n       Replay HTML: {replay_html}"
        )
        return

    # Campaign mode: play sequentially through stages and self-train right after every single stage!
    stages = splits["train"][: args.limit_stages]
    print(f"=== Continual Campaign: Playing & Self-Training across {len(stages)} stages ===")
    t0 = time.perf_counter()
    zs_wins = 0
    post_wins = 0
    for idx, (_, arena, _) in enumerate(stages, 1):
        summary = trainer.play_and_self_train_stage(
            arena, ppo_rounds=args.ppo_rounds, envs=args.envs, horizon=36, record_frames=False
        )
        zs_wins += int(summary["zero_shot_won"])
        post_wins += int(summary["after_train_won"])
        print(
            f"[{idx:02d}/{len(stages)}] {arena.name:<5} | "
            f"Zero-Shot: {'WIN ' if summary['zero_shot_won'] else 'FAIL'} ({summary['zero_shot_outcome']:<5}) -> "
            f"Post-Train: {'WIN ' if summary['after_train_won'] else 'FAIL'} ({summary['after_train_steps']:>3} steps) | "
            f"Memory: {summary['mastered_stages_in_memory']} maps ({summary['elapsed_sec']}s)"
        )

    # Check retention across all played stages + generalization on unseen test maps!
    retention = trainer.evaluate_split(stages)
    unseen_test = trainer.evaluate_split(splits["test"])
    out_ckpt = args.out / "worm_brain_campaign.pt"
    meta = {
        "stages_played": len(stages),
        "zero_shot_wins_during_campaign": zs_wins,
        "post_stage_wins": post_wins,
        "retention_after_campaign": retention,
        "unseen_test_zero_shot": unseen_test,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    save_brain_checkpoint(out_ckpt, trainer.model, trainer.optimizer, metadata=meta)
    print(
        f"\n=== Campaign Complete ({meta['elapsed_sec']}s) ===\n"
        f"  Post-Stage Mastery: {post_wins}/{len(stages)}\n"
        f"  Retention on Campaign Maps: {retention['wins']}/{retention['maps']} ({retention['win_rate']:.1%})\n"
        f"  Unseen Test Split Zero-Shot: {unseen_test['wins']}/{unseen_test['maps']} ({unseen_test['win_rate']:.1%})\n"
        f"  Saved Checkpoint -> {out_ckpt}"
    )


if __name__ == "__main__":
    main()
