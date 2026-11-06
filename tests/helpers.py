"""Shared test helpers: fixture loader and an independent random level sampler."""
import json
import random
from pathlib import Path

from worm_core import GameError, make_level, level_from_dict

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "levels" / "demo-5x5-001.json"


def demo_level():
    return level_from_dict(json.loads(DEMO.read_text(encoding="utf-8")))


def random_level(rng: random.Random, size=None, max_apples=4, wall_frac=0.2):
    """Sample a random *valid* level (not necessarily solvable)."""
    while True:
        n = size or rng.randint(5, 8)
        cells = [(x, y) for x in range(n) for y in range(n)]
        seg_len = rng.randint(2, 5)
        segs = [rng.choice(cells)]
        ok = True
        while len(segs) < seg_len:
            x, y = segs[-1]
            nbrs = [(x + dx, y + dy) for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0))]
            nbrs = [c for c in nbrs if 0 <= c[0] < n and 0 <= c[1] < n and c not in segs]
            if not nbrs:
                ok = False
                break
            segs.append(rng.choice(nbrs))
        if not ok:
            continue
        free = [c for c in cells if c not in segs]
        rng.shuffle(free)
        k = rng.randint(1, max_apples)
        apples, ex = free[:k], free[k]
        rest = free[k + 1:]
        walls = rest[: int(len(cells) * wall_frac * rng.random())]
        try:
            return make_level(n, n, walls, segs, apples, ex)
        except GameError:
            continue
