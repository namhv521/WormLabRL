"""Content hash and symmetry-canonical hash (SPEC-03 §5.1)."""
from __future__ import annotations

import hashlib
import json
from typing import Callable, List

from .rules import Coord, Level

# 8 symmetries of a square board of size n: 4 rotations + 4 reflections.
def _transforms(n: int) -> List[Callable[[Coord], Coord]]:
    m = n - 1
    return [
        lambda c: (c[0], c[1]),
        lambda c: (m - c[1], c[0]),          # rot 90
        lambda c: (m - c[0], m - c[1]),      # rot 180
        lambda c: (c[1], m - c[0]),          # rot 270
        lambda c: (m - c[0], c[1]),          # mirror x
        lambda c: (c[0], m - c[1]),          # mirror y
        lambda c: (c[1], c[0]),              # transpose
        lambda c: (m - c[1], m - c[0]),      # anti-transpose
    ]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content(level: Level, f: Callable[[Coord], Coord]) -> dict:
    return {
        "ruleset_id": level.ruleset_id,
        "grid": {"width": level.width, "height": level.height},
        "walls": sorted([list(f(c)) for c in level.walls]),
        "worm": [list(f(c)) for c in level.segments],   # order head->tail kept
        "apples": sorted([list(f(c)) for c in level.apples]),
        "exit": list(f(level.exit)),
    }


def _sha(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def level_hash(level: Level) -> str:
    return _sha(_content(level, lambda c: c))


def canonical_hash(level: Level) -> str:
    return min(_sha(_content(level, f)) for f in _transforms(level.width))


def transformed_levels(level: Level):
    """All 8 symmetric variants (used by tests)."""
    from dataclasses import replace
    for f in _transforms(level.width):
        yield replace(level, walls=frozenset(f(c) for c in level.walls),
                      segments=tuple(f(c) for c in level.segments),
                      apples=frozenset(f(c) for c in level.apples), exit=f(level.exit))
