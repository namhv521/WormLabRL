"""Seeded candidate construction (SPEC-03 §3.1).

A random *witness walk* of the worm head is simulated first; apples are dropped
on cells the head visits and the exit is placed one step past the walk's end.
Walls go only on cells the witness never touched. The witness is a generation
hint, never a certificate: every candidate must still pass BFS + replay.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from worm_core import DELTAS, GameError, Level, make_level

GENERATOR_VERSION = "witness-walk-v1"
MAX_WALL_FRACTION = 0.25


@dataclass(frozen=True)
class TierConfig:
    sizes: Tuple[int, int]
    segments: Tuple[int, int]
    apples: Tuple[int, int]
    walk: Tuple[int, int]
    wall_frac: Tuple[float, float]


TIER_CONFIGS: Dict[str, TierConfig] = {
    "easy": TierConfig(sizes=(5, 6), segments=(2, 3), apples=(1, 2), walk=(3, 10), wall_frac=(0.0, 0.12)),
    "medium": TierConfig(sizes=(6, 7), segments=(3, 4), apples=(2, 3), walk=(10, 26), wall_frac=(0.08, 0.20)),
    "hard": TierConfig(sizes=(8, 8), segments=(5, 5), apples=(4, 4), walk=(40, 80), wall_frac=(0.20, 0.25)),
}


def candidate_seed(namespace: str, tier: str, base_seed: int, index: int) -> int:
    """Stable uint32 seed per candidate; independent of process/hash randomization."""
    digest = hashlib.sha256(f"{GENERATOR_VERSION}|{namespace}|{tier}|{base_seed}|{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _neighbours(c, n):
    for dx, dy in DELTAS:
        x, y = c[0] + dx, c[1] + dy
        if 0 <= x < n and 0 <= y < n:
            yield (x, y)


def _random_body(rng: random.Random, n: int, length: int):
    cells = [rng.randrange(n), rng.randrange(n)]
    segs = [tuple(cells)]
    while len(segs) < length:
        opts = [c for c in _neighbours(segs[-1], n) if c not in segs]
        if not opts:
            return None
        segs.append(rng.choice(opts))
    return segs


def generate_candidate(tier: str, seed: int) -> Optional[Level]:
    """Build one candidate from ``seed``; returns None when construction fails."""
    cfg = TIER_CONFIGS[tier]
    rng = random.Random(seed)
    n = rng.randint(*cfg.sizes)
    body = _random_body(rng, n, rng.randint(*cfg.segments))
    if body is None:
        return None
    k = rng.randint(*cfg.apples)
    walk_len = max(rng.randint(*cfg.walk), k)
    eat_at = set(rng.sample(range(1, walk_len + 1), k))

    segs = list(body)
    start = set(body)
    touched = set(body)
    apples = []
    prev = None
    for t in range(1, walk_len + 1):
        eating = t in eat_at
        blocked = set(segs) if eating else set(segs[:-1])
        opts = [c for c in _neighbours(segs[0], n) if c not in blocked and c != segs[1]]
        if eating:
            opts = [c for c in opts if c not in start and c not in apples]
        if not opts:
            return None
        # Mild momentum keeps walks from dithering in place.
        straight = None if prev is None else (segs[0][0] + prev[0], segs[0][1] + prev[1])
        nh = straight if straight in opts and rng.random() < 0.35 else rng.choice(opts)
        prev = (nh[0] - segs[0][0], nh[1] - segs[0][1])
        segs = [nh] + (segs if eating else segs[:-1])
        touched.add(nh)
        if eating:
            apples.append(nh)

    exit_opts = [c for c in _neighbours(segs[0], n)
                 if c not in segs[:-1] and c != segs[1] and c not in start and c not in apples]
    if not exit_opts:
        return None
    ex = rng.choice(exit_opts)
    touched.add(ex)

    free = sorted(c for x in range(n) for y in range(n) if (c := (x, y)) not in touched)
    rng.shuffle(free)
    n_walls = min(int(n * n * rng.uniform(*cfg.wall_frac)), int(n * n * MAX_WALL_FRACTION), len(free))
    try:
        return make_level(n, n, free[:n_walls], body, apples, ex, seed=seed,
                          generator_version=GENERATOR_VERSION)
    except GameError:
        return None
