"""Algorithmic difficulty proxy (SPEC-03 §4). Not a human difficulty rating."""
from __future__ import annotations

import math

DIFFICULTY_VERSION = "difficulty-v1"
TIERS = ("easy", "medium", "hard")
BANDS = {"easy": (0.0, 0.35), "medium": (0.35, 0.65), "hard": (0.65, 1.0)}


def difficulty_score(optimal_steps: int, expanded_nodes: int) -> float:
    s = min(optimal_steps / 40.0, 1.0)
    e = min(math.log10(expanded_nodes + 1) / 5.0, 1.0)
    return round(0.60 * s + 0.40 * e, 4)


def tier_of(score: float) -> str:
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score {score} out of [0,1]")
    if score < BANDS["easy"][1]:
        return "easy"
    if score < BANDS["medium"][1]:
        return "medium"
    return "hard"
