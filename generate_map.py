"""
PCGRL Solvable Map Generator CLI with Progressive Escalating Difficulty.

Examples:
  # 1) Generate 5 progressively harder solvable maps (Tier 1 -> Tier 5):
  python generate_map.py --escalating --count 5 --out generated_maps/escalating_pack.json

  # 2) Generate 4 maps specifically at Hard/Extreme difficulty (Tier 4):
  python generate_map.py --tier 4 --count 4 --out generated_maps/tier4_pack.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from train_product import DEFAULT_DESIGNER_CKPT
from worm_ai.pcgrl import (
    DesignerPolicyNet,
    ProgressivePCGRLGenerator,
    load_designer_checkpoint,
    save_designer_checkpoint,
)
from worm_ai.physics import DEFAULT_LEVELS_JSON, ROOT


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Generate 100% solvable Hungry Worm maps via PCGRL")
    parser.add_argument("--designer-ckpt", type=Path, default=DEFAULT_DESIGNER_CKPT)
    parser.add_argument("--train-designer", action="store_true", help="Train PCGRL Designer with PPO first")
    parser.add_argument("--tier", type=int, default=2, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--escalating", action="store_true", help="Monotonically increase difficulty every map!")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "generated_maps" / "escalating_pack.json")
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    torch.set_num_threads(4)
    archive_path = ROOT / "generated_maps" / "archive.json"

    if args.designer_ckpt.exists() and not args.train_designer:
        model, _ = load_designer_checkpoint(args.designer_ckpt)
        print(f"[Loaded] PCGRL Designer from {args.designer_ckpt}")
    else:
        model = DesignerPolicyNet()

    gen = ProgressivePCGRLGenerator(model=model, archive_path=archive_path, seed=args.seed)

    if args.train_designer:
        stats = gen.train_designer_ppo(iterations=20, episodes_per_iter=6, max_edits=16)
        save_designer_checkpoint(args.designer_ckpt, gen.model, metadata=stats)
        print(f"[Saved] Trained PCGRL Designer -> {args.designer_ckpt}")

    base_elements = json.loads(DEFAULT_LEVELS_JSON.read_text(encoding="utf-8-sig"))["elements"]
    generated_levels = []
    floor_score = 10.0
    t0 = time.perf_counter()

    for i in range(args.count):
        cur_tier = min(5, 1 + int(i * 5 / max(1, args.count))) if args.escalating else args.tier
        lv, ores = gen.generate_verified_map(
            target_tier=cur_tier,
            min_difficulty_floor=floor_score if args.escalating else 0.0,
            max_attempts=50,
        )
        if args.escalating:
            floor_score = max(floor_score + 4.0, ores.difficulty_score + 1.5)
        lv["id"] = i + 1
        lv["oracle"] = ores.to_dict()
        generated_levels.append(lv)
        print(
            f"[Map {i + 1:02d}/{args.count}] {lv['name']} | Tier {ores.difficulty_tier} ({ores.tier_name:<11}) | "
            f"Score={ores.difficulty_score:>4.1f} | L*={ores.optimal_steps:>2d} steps | "
            f"Mand(🍎={ores.mandatory_apples}, 🪨={ores.mandatory_stones}) | Deceptive={ores.is_deceptive}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "elements": base_elements,
        "levelsA": generated_levels,
        "levelsB": [],
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[Exported] {len(generated_levels)} verified solvable maps -> {args.out} ({time.perf_counter() - t0:.2f}s)")


if __name__ == "__main__":
    main()
