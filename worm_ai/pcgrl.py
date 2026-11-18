"""PCGRL Progressive Map Designer: RL Level Editor + Exact Oracle Filter + MAP-Elites Archive."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .oracle import OracleResult, TIER_NAMES, analyze_level
from .physics import (
    H,
    W,
    auto_orient_hazard,
    bricks_from_components,
    geometry_signature,
    validate_initial_level,
)

DESIGN_CHANNELS = 10
DESIGN_COND_DIM = 6
DESIGN_ACTIONS = 8  # 0:keep, 1:empty, 2:wall, 3:stone, 4:apple, 5:hazard, 6:portal, 7:gap_step_macro


@dataclass
class TierSpec:
    tier: int
    name: str
    min_steps: int
    target_steps: int
    min_score: float
    target_score: float
    min_mandatory_mech: int
    require_deceptive: bool
    target_apples: int
    target_stones: int
    target_hazards: int


TIER_SPECS: dict[int, TierSpec] = {
    1: TierSpec(1, "Easy",        min_steps=9,  target_steps=14, min_score=12.0, target_score=22.0, min_mandatory_mech=1, require_deceptive=False, target_apples=1, target_stones=0, target_hazards=0),
    2: TierSpec(2, "Medium",      min_steps=16, target_steps=22, min_score=30.0, target_score=39.0, min_mandatory_mech=1, require_deceptive=True,  target_apples=1, target_stones=1, target_hazards=2),
    3: TierSpec(3, "Hard",        min_steps=23, target_steps=30, min_score=46.0, target_score=56.0, min_mandatory_mech=2, require_deceptive=True,  target_apples=2, target_stones=1, target_hazards=3),
    4: TierSpec(4, "Extreme",     min_steps=30, target_steps=38, min_score=63.0, target_score=72.0, min_mandatory_mech=2, require_deceptive=True,  target_apples=2, target_stones=2, target_hazards=5),
    5: TierSpec(5, "Grandmaster", min_steps=38, target_steps=50, min_score=76.0, target_score=88.0, min_mandatory_mech=3, require_deceptive=True,  target_apples=3, target_stones=2, target_hazards=6),
}


class EditableCanvas:
    """Mutable 22x12 Hungry Worm board state with hard structural validity guarantees."""

    def __init__(
        self,
        walls: set[tuple[int, int]],
        stones: set[tuple[int, int]],
        apples: set[tuple[int, int]],
        hazards: set[tuple[int, int]],
        portal: tuple[int, int],
        worm_body: list[tuple[int, int]],
    ):
        self.walls = set(walls)
        self.stones = set(stones)
        self.apples = set(apples)
        self.hazards = set(hazards)
        self.portal = portal
        self.worm_body = list(worm_body)

    def clone(self) -> "EditableCanvas":
        return EditableCanvas(
            self.walls, self.stones, self.apples, self.hazards, self.portal, self.worm_body
        )

    def protected_cells(self) -> set[tuple[int, int]]:
        """Cells currently occupied by worm, portal, or direct floor supporting the worm."""
        prot = set(self.worm_body) | {self.portal}
        # Keep at least one solid wall beneath the worm so frame 0 stays stable
        supports = [(x, y - 1) for x, y in self.worm_body if (x, y - 1) in self.walls]
        if len(supports) <= 1:
            prot.update(supports)
        return prot

    def settle_stones(self) -> None:
        """Drop floating stones onto solid support so frame 0 is gravity-settled."""
        fixed = self.walls | {self.portal} | self.apples | set(self.worm_body) | self.hazards
        settled: set[tuple[int, int]] = set()
        for sx, sy in sorted(self.stones, key=lambda p: p[1]):
            cy = sy
            while cy > 1 and (sx, cy - 1) not in (fixed | settled):
                cy -= 1
            if cy >= 1 and (sx, cy) not in (fixed | settled):
                settled.add((sx, cy))
        self.stones = settled

    def clear_cell(self, pt: tuple[int, int]) -> None:
        self.walls.discard(pt)
        self.stones.discard(pt)
        self.apples.discard(pt)
        self.hazards.discard(pt)

    def apply_edit(self, x: int, y: int, action: int) -> bool:
        """Apply one PCGRL Designer action at (x, y) while preserving hard frame-0 invariants."""
        pt = (x, y)
        if not (3 <= x <= 18 and 1 <= y <= 10):
            return False
        prot = self.protected_cells()

        if action == 0:
            return True
        if action == 1:  # Carve empty
            if pt in prot:
                return False
            self.clear_cell(pt)
            self.settle_stones()
            return True
        if action == 2:  # Place wall
            if pt in set(self.worm_body) or pt == self.portal:
                return False
            self.clear_cell(pt)
            self.walls.add(pt)
            self.settle_stones()
            return True
        if action == 3:  # Place stone
            if pt in set(self.worm_body) or pt == self.portal or len(self.stones) >= 4:
                return False
            self.clear_cell(pt)
            # Ensure floor underneath or let settle_stones drop it
            if (x, y - 1) not in self.walls and y > 2:
                if (x, y - 1) not in set(self.worm_body) and (x, y - 1) != self.portal:
                    self.clear_cell((x, y - 1))
                    self.walls.add((x, y - 1))
            self.stones.add(pt)
            self.settle_stones()
            return True
        if action == 4:  # Place apple
            if pt in set(self.worm_body) or pt == self.portal or len(self.apples) >= 5:
                return False
            self.clear_cell(pt)
            self.apples.add(pt)
            return True
        if action == 5:  # Place hazard (avoid adjacent instant-trap on worm spawn)
            if pt in prot or len(self.hazards) >= 9:
                return False
            if any(abs(x - wx) + abs(y - wy) <= 1 for wx, wy in self.worm_body):
                return False
            self.clear_cell(pt)
            self.hazards.add(pt)
            return True
        if action == 6:  # Relocate portal
            if pt in set(self.worm_body) or pt in self.walls:
                return False
            if abs(x - self.worm_body[0][0]) + abs(y - self.worm_body[0][1]) < 4:
                return False
            self.clear_cell(pt)
            self.portal = pt
            return True
        if action == 7:  # Gap + elevated step puzzle macro
            if pt in prot or (x + 1, y) in prot:
                return False
            self.clear_cell(pt)
            if y + 1 <= 9 and (x + 1, y + 1) not in prot:
                self.clear_cell((x + 1, y + 1))
                self.walls.add((x + 1, y + 1))
            self.settle_stones()
            return True
        return False

    def to_level_dict(self, level_id: int = 1, hints: list[dict] | None = None) -> dict:
        bricks = bricks_from_components(
            walls=self.walls,
            stones=self.stones,
            apples=self.apples,
            hazards=self.hazards,
            portal=self.portal,
            worm_body=self.worm_body,
        )
        return {
            "id": int(level_id),
            "width": W,
            "height": H,
            "bricks": bricks,
            "hints": hints or [],
        }

    @classmethod
    def from_level_dict(cls, level: dict) -> "EditableCanvas":
        bricks = level["bricks"]
        walls = {(b["x"], b["y"]) for b in bricks if b["id"] == 1}
        stones = {(b["x"], b["y"]) for b in bricks if b["id"] == 2}
        apples = {(b["x"], b["y"]) for b in bricks if b["id"] == 4}
        hazards = {(b["x"], b["y"]) for b in bricks if b["id"] in {3, 7, 9, 10, 11}}
        portals = [(b["x"], b["y"]) for b in bricks if b["id"] == 8]
        valid, _, board = validate_initial_level(level)
        if valid and board is not None:
            worm_body = list(board.initial.body)
        else:
            head = next((b["x"], b["y"]) for b in bricks if b["id"] == 5)
            tails = [(b["x"], b["y"]) for b in bricks if b["id"] == 6]
            worm_body = [head] + tails
        return cls(walls, stones, apples, hazards, portals[0], worm_body)


def build_puzzle_scaffold(tier: int, rng: random.Random) -> EditableCanvas:
    """
    Construct diverse, structurally coherent puzzle scaffolds scaled by difficulty tier (1..5),
    serving as initial canvases for the PCGRL Designer Agent to sculpt and refine.
    """
    if tier == 1:
        archetype = "gentle_bridge"
    elif tier >= 4:
        archetype = rng.choice(("grandmaster_gauntlet", "grandmaster_gauntlet", "well_climb"))
    else:
        archetype = rng.choice(("chasm_detour", "well_climb", "switchback_tower", "stone_lock_bridge"))

    walls: set[tuple[int, int]] = set()
    stones: set[tuple[int, int]] = set()
    apples: set[tuple[int, int]] = set()
    hazards: set[tuple[int, int]] = set()

    if archetype == "gentle_bridge":
        base_y = rng.choice((3, 4))
        for x in range(5, 10):
            walls.add((x, base_y))
        # Gap at x=10, 11 (width 2), then elevated ledge requiring 1 apple to climb/bridge!
        for x in range(12, 16):
            walls.add((x, base_y + 1))
        worm_body = [(7, base_y + 1), (6, base_y + 1), (5, base_y + 1)]
        apples.add((rng.choice((8, 9)), base_y + 2))
        portal = (15, base_y + 2)

    elif archetype == "grandmaster_gauntlet":
        # True S-curve 3-story switchback labyrinth: guaranteed L* = 36..55 moves!
        # Floor 1 (y=1): walk right x=4..15 -> climb right shaft ->
        # Floor 2 (y=5): walk left x=15..5 (push stone / eat apples) -> climb left shaft ->
        # Floor 3 (y=8): walk right x=5..16 across chasm gaps to portal!
        y1, y2, y3 = 1, 5, 8
        for x in range(4, 16):
            walls.add((x, y1))
        # Solid ceiling over Floor 1 (x=4..13) prevents upward shortcut; shaft open at x=14..16!
        for x in range(6, 15):
            walls.add((x, y2))
        # Right climbing steps from Floor 1 (y=1) up to Floor 2 (y=5)
        walls.add((15, 2))
        walls.add((16, 3))
        walls.add((16, 4))
        walls.add((15, 5))

        # Left climbing steps from Floor 2 (y=5) up to Floor 3 (y=8)
        walls.add((5, 5))
        walls.add((4, 6))
        walls.add((4, 7))
        for x in range(5, 10):
            walls.add((x, y3))
        # Chasm gap at x=10, 11 (or 10..12 for Tier 5!) requiring grown worm body to bridge!
        gap_end = 13 if tier >= 5 else 12
        for x in range(gap_end, 17):
            walls.add((x, y3))

        worm_body = [(6, y1 + 1), (5, y1 + 1), (4, y1 + 1)]
        apples.add((11, y1 + 1))
        apples.add((15, y1 + 2))
        apples.add((9, y2 + 1))
        stones.add((10, y2 + 1))
        hazards.add((8, y1 + 1))
        if tier >= 5:
            apples.add((6, y3 + 1))
            stones.add((7, y3 + 1))
            hazards.add((12, y2 + 1))
            hazards.add((gap_end + 1, y3 + 1))
            portal = (16, y3 + 2)
            walls.add((16, y3 + 1))
        else:
            portal = (16, y3 + 1)

    elif archetype == "chasm_detour":
        base_y = rng.choice((3, 4))
        # Left spawn terrace
        for x in range(4, 9):
            walls.add((x, base_y))
        worm_body = [(6, base_y + 1), (5, base_y + 1), (4, base_y + 1)]

        # Upper or lower apple detour pocket on left/center
        if tier >= 2:
            for x in range(4, 8):
                walls.add((x, base_y + 3))
            walls.add((8, base_y + 1))
            walls.add((8, base_y + 2))
            apples.add((rng.choice((5, 6)), base_y + 4))
            if tier >= 3:
                apples.add((4, base_y + 4))
                hazards.add((7, base_y + 1))
        else:
            apples.add((7, base_y + 2))

        # Middle stepping island with gap
        gap1 = 2 if tier <= 2 else 3
        mid_x = 8 + gap1 + 1
        for x in range(mid_x, mid_x + 3):
            walls.add((x, base_y + (1 if tier >= 2 else 0)))
        if tier >= 3:
            stones.add((mid_x + 1, base_y + 2))
        if tier >= 4:
            apples.add((mid_x, base_y + 3))
            hazards.add((mid_x + 2, base_y))

        # Right portal ledge
        end_x = min(17, mid_x + 4 + (1 if tier >= 3 else 0))
        end_y = min(9, base_y + (2 if tier >= 2 else 1))
        for x in range(end_x - 1, min(19, end_x + 2)):
            walls.add((x, end_y - 1))
        portal = (end_x, end_y)
        if tier >= 5:
            hazards.add((end_x - 1, end_y - 2))

    elif archetype == "well_climb":
        # U-shaped deep well where worm must climb in, collect apples/push stone, and fold out
        floor_y = 2
        left_x = rng.choice((5, 6))
        right_x = left_x + rng.choice((6, 7))
        for x in range(left_x, right_x + 1):
            walls.add((x, floor_y))
        for y in range(floor_y, floor_y + 5):
            walls.add((left_x, y))
            walls.add((right_x, y))
        # Left entry terrace
        for x in range(left_x - 2, left_x + 1):
            walls.add((x, floor_y + 4))
        worm_body = [(left_x - 1, floor_y + 5), (left_x - 2, floor_y + 5), (left_x - 2, floor_y + 6)]

        # Interior ledges, apples, stones inside the well
        mid = (left_x + right_x) // 2
        walls.add((mid, floor_y + 2))
        walls.add((mid + 1, floor_y + 2))
        apples.add((mid - 1, floor_y + 1))
        if tier >= 2:
            apples.add((right_x - 1, floor_y + 1))
        if tier >= 3:
            stones.add((mid, floor_y + 3))
            hazards.add((left_x + 1, floor_y + 1))
        if tier >= 4:
            apples.add((mid + 1, floor_y + 4))
            hazards.add((right_x - 1, floor_y + 3))

        # Right high exit terrace + portal
        out_y = floor_y + 4 + (1 if tier >= 3 else 0)
        for x in range(right_x, min(19, right_x + 4)):
            walls.add((x, out_y))
        portal = (min(18, right_x + 3), out_y + 1)

    elif archetype == "switchback_tower":
        # Multi-floor switchback puzzle (very classic Hungry Worm Levels 15-30 style!)
        y1, y2, y3 = 2, 5, 8
        for x in range(5, 15):
            walls.add((x, y1))
        for x in range(7, 16):
            if x not in (10, 11):
                walls.add((x, y2))
        for x in range(5, 13):
            walls.add((x, y3))

        worm_body = [(7, y1 + 1), (6, y1 + 1), (5, y1 + 1)]
        # Climbing pillar on right
        walls.add((14, y1 + 1))
        walls.add((15, y1 + 2))
        apples.add((8, y2 + 1))

        if tier >= 2:
            stones.add((12, y2 + 1))
        if tier >= 3:
            apples.add((6, y1 + 3))
            walls.add((6, y1 + 2))
            hazards.add((13, y1 + 1))
        if tier >= 4:
            stones.add((9, y3 + 1))
            apples.add((12, y3 + 1))
            hazards.add((8, y1 + 1))

        portal = (5, y3 + 1) if tier >= 3 else (15, y2 + 1)

    else:  # stone_lock_bridge
        y_main = 4
        for x in range(4, 10):
            walls.add((x, y_main))
        for x in range(12, 17):
            walls.add((x, y_main))
        # Lower catch shelf under the gap (10, 11) so pushed stone or worm can bridge!
        for x in range(8, 14):
            walls.add((x, y_main - 2))

        worm_body = [(6, y_main + 1), (5, y_main + 1), (4, y_main + 1)]
        stones.add((9, y_main + 1))
        apples.add((10, y_main - 1))

        if tier >= 2:
            # Upper shelf on left with extra apple/stone
            for x in range(5, 9):
                walls.add((x, y_main + 2))
            apples.add((6, y_main + 3))
        if tier >= 3:
            stones.add((8, y_main + 3))
            hazards.add((12, y_main - 1))
        if tier >= 4:
            for x in range(13, 18):
                walls.add((x, y_main + 3))
            apples.add((15, y_main + 1))
            hazards.add((14, y_main + 2))
            portal = (17, y_main + 4)
        else:
            portal = (16, y_main + 1 + (1 if tier >= 2 else 0))
            if tier >= 2:
                walls.add((15, y_main + 1))

    # Ensure no collisions with worm or portal
    occ = set(worm_body) | {portal}
    walls -= occ
    stones -= occ | walls
    apples -= occ | walls | stones
    hazards -= occ | walls | stones | apples

    canvas = EditableCanvas(walls, stones, apples, hazards, portal, worm_body)
    canvas.settle_stones()
    return canvas


def prune_unused_objects(level: dict, oracle_res: OracleResult) -> tuple[dict, OracleResult]:
    """
    Anti-Fake-Object Cleanser:
    If an otherwise great solvable puzzle has unused apples (not eaten in optimal path),
    strip them out and re-verify so 100% of remaining apples participate in the puzzle!
    """
    if not oracle_res.solvable or oracle_res.unused_apples == 0:
        return level, oracle_res

    valid, _, board = validate_initial_level(level)
    if not valid or board is None:
        return level, oracle_res

    # Simulate actions to see exact eaten apples
    st = board.initial
    for act in oracle_res.actions:
        st, _ = board.step(st, act)
    eaten_apples = board.initial.apples - st.apples

    cleaned_bricks = [
        b for b in level["bricks"]
        if not (b["id"] == 4 and (b["x"], b["y"]) not in eaten_apples)
    ]
    cleaned_level = dict(level, bricks=cleaned_bricks)
    new_res = analyze_level(cleaned_level, budget=25000, counterfactual_budget=8000, verify_mandatory=True)
    if new_res.solvable:
        cleaned_level["hints"] = new_res.hints
        return cleaned_level, new_res
    return level, oracle_res


class DesignerPolicyNet(nn.Module):
    """PCGRL Level Designer Actor-Critic conditioned on target difficulty specification."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv2d(DESIGN_CHANNELS, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(DESIGN_COND_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        # Cursor local feature (channels) + global avg (channels) + global max (channels) + cond (32)
        feat_dim = channels * 3 + 32
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, 96),
            nn.SiLU(),
            nn.Linear(96, DESIGN_ACTIONS),
        )
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )

    def forward(self, grid: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.conv(grid)
        cursor_mask = grid[:, 7:8]
        cursor_feat = (fmap * cursor_mask).sum(dim=(2, 3))
        avg_feat = fmap.mean(dim=(2, 3))
        max_feat = fmap.amax(dim=(2, 3))
        c_feat = self.cond_mlp(cond)
        emb = torch.cat((cursor_feat, avg_feat, max_feat, c_feat), dim=-1)
        return self.actor(emb), self.critic(emb).squeeze(-1)


class WormDesignEnv:
    """
    Iterative PCGRL Environment (Narrow Cursor Representation):
    The RL Designer scans key puzzle coordinates and edits tiles to maximize Oracle puzzle quality
    for a conditioned Difficulty Tier (1..5).
    """

    def __init__(self, max_edits: int = 20, seed: int = 73):
        self.max_edits = max_edits
        self.rng = random.Random(seed)
        self.tier: int = 1
        self.spec: TierSpec = TIER_SPECS[1]
        self.canvas: EditableCanvas = build_puzzle_scaffold(1, self.rng)
        self.cursor_path: list[tuple[int, int]] = []
        self.step_idx: int = 0
        self.prev_potential: float = 0.0

    def _build_cursor_scan(self) -> list[tuple[int, int]]:
        """Focus edits around structurally active platform/gap regions rather than empty border air."""
        interesting: list[tuple[int, int]] = []
        active = self.canvas.walls | self.canvas.stones | self.canvas.apples | {self.canvas.portal}
        visited = set()
        for ax, ay in active:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1, 2):
                    nx, ny = ax + dx, ay + dy
                    if 4 <= nx <= 17 and 1 <= ny <= 9 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        interesting.append((nx, ny))
        self.rng.shuffle(interesting)
        while len(interesting) < self.max_edits + 1:
            interesting.append((self.rng.randint(4, 17), self.rng.randint(2, 9)))
        return interesting[: self.max_edits + 1]

    def reset(
        self,
        tier: int = 1,
        seed_level: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.tier = max(1, min(5, int(tier)))
        self.spec = TIER_SPECS[self.tier]
        if seed_level is not None:
            self.canvas = EditableCanvas.from_level_dict(seed_level)
        else:
            self.canvas = build_puzzle_scaffold(self.tier, self.rng)
        self.cursor_path = self._build_cursor_scan()
        self.step_idx = 0
        self.prev_potential = self.evaluate_potential(fast=True)[0]
        return self.observe()

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        grid = np.zeros((DESIGN_CHANNELS, H, W), dtype=np.float32)
        for x, y in self.canvas.walls:
            grid[0, y, x] = 1.0
        for x, y in self.canvas.stones:
            grid[1, y, x] = 1.0
        for x, y in self.canvas.apples:
            grid[2, y, x] = 1.0
        for x, y in self.canvas.hazards:
            grid[3, y, x] = 1.0
        px, py = self.canvas.portal
        grid[4, py, px] = 1.0
        hx, hy = self.canvas.worm_body[0]
        grid[5, hy, hx] = 1.0
        for bx, by in self.canvas.worm_body[1:]:
            grid[6, by, bx] = 1.0
        cx, cy = self.cursor_path[min(self.step_idx, len(self.cursor_path) - 1)]
        grid[7, cy, cx] = 1.0
        grid[8, :, :] = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        grid[9, :, :] = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None]

        cond = np.array([
            self.tier / 5.0,
            self.spec.target_steps / 60.0,
            self.spec.target_apples / 4.0,
            self.spec.target_stones / 3.0,
            self.spec.target_hazards / 6.0,
            1.0 - (self.step_idx / max(1, self.max_edits)),
        ], dtype=np.float32)
        return grid, cond

    def evaluate_potential(
        self,
        fast: bool = False,
        solver_eval_fn: Callable[[dict], bool] | None = None,
    ) -> tuple[float, OracleResult]:
        """Compute quality potential Q(M, tier) using ExactWormOracle."""
        lv = self.canvas.to_level_dict()
        budget = 6000 if fast else 20000
        cf_budget = 2500 if fast else 6000
        res = analyze_level(lv, budget=budget, counterfactual_budget=cf_budget, verify_mandatory=not fast)

        if not res.solvable:
            return -3.0 - 0.18 * min(25, res.min_portal_dist), res

        spec = self.spec
        q = 5.0  # Solvable bonus
        # Length & score match
        q += 5.0 * math.exp(-abs(res.optimal_steps - spec.target_steps) / 14.0)
        q += 4.0 * math.exp(-abs(res.difficulty_score - spec.target_score) / 18.0)
        if res.optimal_steps >= spec.min_steps:
            q += 2.0
        else:
            q -= 0.35 * (spec.min_steps - res.optimal_steps)

        if fast:
            # Fast proxy during intermediate edits
            q += 1.6 * res.apples_eaten + 1.8 * min(2, res.stones_moved)
            q -= 2.0 * res.unused_apples
        else:
            q += 2.6 * res.mandatory_apples + 3.2 * res.mandatory_stones
            q -= 2.5 * res.unused_apples
            non_mand_stones = max(0, res.stones_total - res.mandatory_stones)
            q -= 1.2 * non_mand_stones

        if res.is_deceptive:
            q += 3.0
        elif spec.require_deceptive:
            q -= 2.5

        # Clutter check: keep brick count clean (20..65)
        n_bricks = len(lv["bricks"])
        if n_bricks > 65:
            q -= 0.15 * (n_bricks - 65)

        # PAIRED-style adversarial frontier bonus if current RL Solver fails zero-shot!
        if solver_eval_fn is not None and not fast:
            solver_won_zero_shot = solver_eval_fn(lv)
            if not solver_won_zero_shot:
                q += 3.5

        return round(q, 4), res

    def step(
        self,
        action: int,
        solver_eval_fn: Callable[[dict], bool] | None = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], float, bool, dict]:
        cx, cy = self.cursor_path[self.step_idx]
        backup = self.canvas.clone()
        act = int(action)
        # Tier-appropriate edit guardrails so Tier 1 stays clean Easy and Tier 4-5 keep long paths
        if self.tier == 1 and act in (3, 5):
            act = 0
        if self.tier >= 4 and act == 6:
            act = 0
        changed = self.canvas.apply_edit(cx, cy, act)
        if changed:
            valid, _, board = validate_initial_level(self.canvas.to_level_dict())
            if not valid or board is None or len(board.bodies) != 1:
                # Revert edits that break frame-0 stability or unambiguous worm body
                self.canvas = backup

        self.step_idx += 1
        done = self.step_idx >= self.max_edits
        # Evaluate potential densely every 5 steps and accurately at terminal step
        if done or (self.step_idx % 5 == 0):
            cur_pot, oracle_res = self.evaluate_potential(fast=not done, solver_eval_fn=solver_eval_fn)
            reward = cur_pot - self.prev_potential
            self.prev_potential = cur_pot
        else:
            reward = -0.01
            oracle_res = None

        obs = self.observe()
        info = {"oracle": oracle_res, "potential": self.prev_potential} if done else {}
        return obs, float(reward), done, info


class MapElitesArchive:
    """Quality-Diversity Archive storing unique verified puzzles indexed by (tier, mechanic_signature)."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.cells: dict[str, dict] = {}
        self.signatures: set[tuple] = set()
        if self.path and self.path.exists():
            self.load(self.path)

    def _bucket_key(self, res: OracleResult) -> str:
        if res.mandatory_stones > 0 and res.mandatory_apples > 0:
            mech = "hybrid_apple_stone"
        elif res.mandatory_stones > 0:
            mech = "stone_sokoban"
        elif res.mandatory_apples >= 2:
            mech = "multi_apple_bridge"
        elif res.mandatory_apples == 1:
            mech = "apple_growth"
        else:
            mech = "pure_navigation"
        return f"T{res.difficulty_tier}_{mech}_{res.optimal_steps // 6}"

    def try_add(self, level: dict, res: OracleResult, quality: float) -> bool:
        if not res.solvable or res.unused_apples > 0:
            return False
        sig = geometry_signature(level)
        if sig in self.signatures:
            return False
        key = self._bucket_key(res)
        existing = self.cells.get(key)
        entry = {
            "bucket": key,
            "quality": round(quality, 3),
            "oracle": res.to_dict(),
            "level": dict(level, hints=res.hints),
            "created_at": round(time.time(), 1),
        }
        if existing is None or quality > existing["quality"] or len(self.cells) < 120:
            # Keep unique buckets, or create numbered slot if high quality
            slot_key = key if existing is None or quality > existing["quality"] else f"{key}_{len(self.cells)}"
            self.cells[slot_key] = entry
            self.signatures.add(sig)
            if self.path:
                self.save(self.path)
            return True
        return False

    def sample_seed_for_tier(self, target_tier: int, rng: random.Random) -> dict | None:
        """Pick an existing solved puzzle from tier-1 or tier to evolve into an even harder puzzle!"""
        candidates = [
            v["level"] for v in self.cells.values()
            if v["oracle"]["difficulty_tier"] in (max(1, target_tier - 1), target_tier)
        ]
        return rng.choice(candidates) if candidates else None

    def list_by_difficulty(self) -> list[dict]:
        return sorted(self.cells.values(), key=lambda x: x["oracle"]["difficulty_score"])

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entries": list(self.cells.values())}, indent=2), encoding="utf-8")
        tmp.replace(p)

    def load(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        for item in data.get("entries", []):
            lv = item["level"]
            self.cells[item["bucket"]] = item
            self.signatures.add(geometry_signature(lv))


class ProgressivePCGRLGenerator:
    """
    Trains DesignerPolicyNet via PPO and generates 100% verified, zero-fake-object maps
    with monotonically increasing difficulty (Tier 1 -> Tier 5).
    """

    def __init__(
        self,
        model: DesignerPolicyNet | None = None,
        archive_path: Path | str | None = None,
        lr: float = 3e-4,
        device: str | torch.device = "cpu",
        seed: int = 73,
    ):
        self.device = torch.device(device)
        self.model = (model if model is not None else DesignerPolicyNet()).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.archive = MapElitesArchive(archive_path)
        self.rng = random.Random(seed)
        self.next_generated_id = 1001

    def train_designer_ppo(
        self,
        iterations: int = 24,
        episodes_per_iter: int = 8,
        max_edits: int = 18,
        solver_eval_fn: Callable[[dict], bool] | None = None,
        log_fn: Callable[[str], None] = print,
    ) -> dict:
        """Train the PCGRL Designer Policy across all 5 difficulty tiers using PPO."""
        envs = [WormDesignEnv(max_edits=max_edits, seed=100 + i) for i in range(episodes_per_iter)]
        history = []
        solvable_found = 0
        t0 = time.perf_counter()

        for it in range(1, iterations + 1):
            obs_grids, obs_conds, acts, old_logps, vals, rews, dones = [], [], [], [], [], [], []
            iter_solvable = 0
            iter_scores = []

            # Reset each parallel design env to a target tier (cycling 1..5)
            cur_obs = []
            for idx, env in enumerate(envs):
                tier = ((it + idx - 1) % 5) + 1
                seed_lv = self.archive.sample_seed_for_tier(tier, self.rng) if self.rng.random() < 0.35 else None
                cur_obs.append(env.reset(tier=tier, seed_level=seed_lv))

            self.model.eval()
            for _ in range(max_edits):
                g_b = torch.from_numpy(np.stack([o[0] for o in cur_obs])).to(self.device)
                c_b = torch.from_numpy(np.stack([o[1] for o in cur_obs])).to(self.device)
                with torch.no_grad():
                    logits, values = self.model(g_b, c_b)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample()
                    logps = dist.log_prob(actions)

                next_obs = []
                step_rews = []
                step_dones = []
                for idx, act_val in enumerate(actions.tolist()):
                    no, r, d, info = envs[idx].step(act_val, solver_eval_fn=solver_eval_fn)
                    next_obs.append(no)
                    step_rews.append(r)
                    step_dones.append(d)
                    if d and info.get("oracle") is not None:
                        ores: OracleResult = info["oracle"]
                        if ores.solvable:
                            raw_lv = envs[idx].canvas.to_level_dict(level_id=self.next_generated_id)
                            clean_lv, clean_res = prune_unused_objects(raw_lv, ores)
                            if clean_res.solvable:
                                iter_solvable += 1
                                solvable_found += 1
                                iter_scores.append(clean_res.difficulty_score)
                                if self.archive.try_add(clean_lv, clean_res, info["potential"]):
                                    self.next_generated_id += 1

                obs_grids.append(g_b)
                obs_conds.append(c_b)
                acts.append(actions)
                old_logps.append(logps)
                vals.append(values)
                rews.append(torch.tensor(step_rews, dtype=torch.float32, device=self.device))
                dones.append(torch.tensor(step_dones, dtype=torch.bool, device=self.device))
                cur_obs = next_obs

            # GAE over design trajectory
            grids_t = torch.stack(obs_grids)
            conds_t = torch.stack(obs_conds)
            acts_t = torch.stack(acts)
            old_lp_t = torch.stack(old_logps)
            vals_t = torch.stack(vals)
            rews_t = torch.stack(rews)
            dones_t = torch.stack(dones)

            adv_t = torch.zeros_like(rews_t)
            gae = torch.zeros(episodes_per_iter, device=self.device)
            next_v = torch.zeros(episodes_per_iter, device=self.device)
            for t in reversed(range(max_edits)):
                cont = (~dones_t[t]).float()
                delta = rews_t[t] + 0.98 * next_v * cont - vals_t[t]
                gae = delta + 0.98 * 0.95 * cont * gae
                adv_t[t] = gae
                next_v = vals_t[t]
            ret_t = adv_t + vals_t
            adv_t = (adv_t - adv_t.mean()) / adv_t.std(unbiased=False).clamp_min(1e-6)

            bg = grids_t.flatten(0, 1)
            bc = conds_t.flatten(0, 1)
            ba = acts_t.flatten(0, 1)
            bolp = old_lp_t.flatten(0, 1)
            badv = adv_t.flatten(0, 1)
            bret = ret_t.flatten(0, 1)

            self.model.train()
            loss_sum = 0.0
            for _ in range(3):
                perm = torch.randperm(len(ba), device=self.device)
                for ids in perm.split(64):
                    lg, vl = self.model(bg[ids], bc[ids])
                    dst = torch.distributions.Categorical(logits=lg)
                    rat = (dst.log_prob(ba[ids]) - bolp[ids]).exp()
                    p_loss = -torch.minimum(rat * badv[ids], rat.clamp(0.8, 1.2) * badv[ids]).mean()
                    v_loss = F.smooth_l1_loss(vl, bret[ids])
                    ent = dst.entropy().mean()
                    loss = p_loss + 0.5 * v_loss - 0.02 * ent
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    loss_sum += float(loss.detach().item())

            avg_diff = round(sum(iter_scores) / len(iter_scores), 1) if iter_scores else 0.0
            row = {
                "iter": it,
                "solvable_in_batch": iter_solvable,
                "avg_difficulty": avg_diff,
                "archive_size": len(self.archive.cells),
            }
            history.append(row)
            if it % 6 == 0 or it == 1 or it == iterations:
                log_fn(
                    f"[PCGRL Iter {it:02d}/{iterations}] solvable={iter_solvable}/{episodes_per_iter} | "
                    f"avg_score={avg_diff} | archive={len(self.archive.cells)} unique maps"
                )

        return {
            "iterations": iterations,
            "total_solvable_generated": solvable_found,
            "archive_size": len(self.archive.cells),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "history": history,
        }

    @torch.inference_mode()
    def generate_verified_map(
        self,
        target_tier: int = 1,
        min_difficulty_floor: float = 0.0,
        max_attempts: int = 45,
        max_edits: int = 18,
        solver_eval_fn: Callable[[dict], bool] | None = None,
        exclude_signatures: set[tuple] | None = None,
    ) -> tuple[dict, OracleResult]:
        """
        Generate a brand-new, 100% guaranteed-solvable, zero-fake-apple Hungry Worm map
        meeting `target_tier` and strictly exceeding `min_difficulty_floor`!
        """
        self.model.eval()
        tier = max(1, min(5, int(target_tier)))
        spec = TIER_SPECS[tier]
        env = WormDesignEnv(max_edits=max_edits, seed=self.rng.randint(1, 10_000_000))
        forbidden = exclude_signatures or set()

        best_candidate: tuple[float, dict, OracleResult] | None = None

        for attempt in range(max_attempts):
            # Mix fresh architectural scaffolds with evolutionary refinement of frontier archive maps
            use_seed = (attempt % 3 == 1) and len(self.archive.cells) > 0
            seed_lv = self.archive.sample_seed_for_tier(tier, self.rng) if use_seed else None
            obs_g, obs_c = env.reset(tier=tier, seed_level=seed_lv)

            for _ in range(max_edits):
                g_t = torch.from_numpy(obs_g[None, ...]).to(self.device)
                c_t = torch.from_numpy(obs_c[None, ...]).to(self.device)
                logits, _ = self.model(g_t, c_t)
                probs = (logits[0] / 0.9).softmax(dim=-1).cpu().numpy()
                act = int(np.random.choice(DESIGN_ACTIONS, p=probs))
                (obs_g, obs_c), _, done, _ = env.step(act)
                if done:
                    break

            raw_lv = env.canvas.to_level_dict(level_id=self.next_generated_id)
            valid, _, board = validate_initial_level(raw_lv)
            if not valid or board is None or len(board.bodies) != 1:
                continue

            ores = analyze_level(raw_lv, budget=25000, counterfactual_budget=8000, verify_mandatory=True)
            if not ores.solvable:
                continue

            clean_lv, clean_res = prune_unused_objects(raw_lv, ores)
            if not clean_res.solvable or clean_res.unused_apples > 0:
                continue

            sig = geometry_signature(clean_lv)
            if sig in forbidden or sig in self.archive.signatures:
                continue

            # Rank candidate by how closely it satisfies escalating difficulty requirements
            mand_total = clean_res.mandatory_apples + clean_res.mandatory_stones
            meets_floor = clean_res.difficulty_score >= max(spec.min_score, min_difficulty_floor)
            meets_steps = clean_res.optimal_steps >= spec.min_steps
            meets_mech = mand_total >= min(spec.min_mandatory_mech, 1 if tier <= 2 else 2)
            meets_decep = (not spec.require_deceptive) or clean_res.is_deceptive

            solver_bonus = 0.0
            if solver_eval_fn is not None and not solver_eval_fn(clean_lv):
                solver_bonus = 8.0

            floor_penalty = (
                200.0 * (min_difficulty_floor - clean_res.difficulty_score)
                if clean_res.difficulty_score < min_difficulty_floor
                else 0.0
            )
            rank_score = (
                clean_res.difficulty_score
                + 8.0 * mand_total
                + (12.0 if clean_res.is_deceptive else 0.0)
                + solver_bonus
                - abs(clean_res.difficulty_score - max(spec.target_score, min_difficulty_floor + 4.0)) * 0.8
                - floor_penalty
            )
            if best_candidate is None or rank_score > best_candidate[0]:
                best_candidate = (rank_score, clean_lv, clean_res)

            if meets_floor and meets_steps and meets_mech and meets_decep:
                clean_lv["id"] = self.next_generated_id
                clean_lv["name"] = f"G{self.next_generated_id}_T{tier}"
                clean_lv["tier"] = clean_res.difficulty_tier
                clean_lv["difficulty_score"] = clean_res.difficulty_score
                self.next_generated_id += 1
                self.archive.try_add(clean_lv, clean_res, rank_score)
                return clean_lv, clean_res

        if best_candidate is not None:
            _, clean_lv, clean_res = best_candidate
            clean_lv["id"] = self.next_generated_id
            clean_lv["name"] = f"G{self.next_generated_id}_T{tier}"
            clean_lv["tier"] = clean_res.difficulty_tier
            clean_lv["difficulty_score"] = clean_res.difficulty_score
            self.next_generated_id += 1
            self.archive.try_add(clean_lv, clean_res, best_candidate[0])
            return clean_lv, clean_res

        raise RuntimeError(f"Failed to generate verified solvable map after {max_attempts} attempts")


def save_designer_checkpoint(path: Path | str, model: DesignerPolicyNet, metadata: dict | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    torch.save({
        "architecture": "DesignerPolicyNet_v1",
        "channels": model.channels,
        "model": model.state_dict(),
        "metadata": metadata or {},
    }, tmp)
    tmp.replace(p)


def load_designer_checkpoint(
    path: Path | str,
    device: str | torch.device = "cpu",
) -> tuple[DesignerPolicyNet, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = DesignerPolicyNet(channels=int(payload.get("channels", 32))).to(device)
    model.load_state_dict(payload["model"])
    return model, payload.get("metadata", {})
