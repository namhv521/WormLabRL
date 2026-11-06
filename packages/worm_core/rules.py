"""Canonical rules for ruleset ``worm-grid-v1`` (SPEC-01).

Every consumer (env, solver, generator, API) must call these functions;
no other module may re-implement move legality or terminal logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Optional, Sequence, Tuple

RULESET_ID = "worm-grid-v1"
RULESET_VERSION = "worm-grid-v1.0"
SCHEMA_VERSION = "1.0"

MIN_SIZE, MAX_SIZE = 5, 8
MIN_SEGMENTS, MAX_SEGMENTS = 2, 5
MIN_APPLES, MAX_APPLES = 1, 4

Coord = Tuple[int, int]

ACTIONS: Tuple[str, ...] = ("UP", "RIGHT", "DOWN", "LEFT")
ACTION_IDS = {name: i for i, name in enumerate(ACTIONS)}
DELTAS: Tuple[Coord, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


class Status(str, Enum):
    PLAYING = "PLAYING"
    WON = "WON"
    LOST = "LOST"
    TIMEOUT = "TIMEOUT"


class GameError(Exception):
    """Error with a stable machine-readable code (see SPEC-05 §4)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Level:
    """Immutable level definition. ``segments`` is ordered head -> tail."""

    width: int
    height: int
    walls: FrozenSet[Coord]
    segments: Tuple[Coord, ...]
    apples: FrozenSet[Coord]
    exit: Coord
    ruleset_id: str = RULESET_ID
    level_id: Optional[str] = None
    seed: Optional[int] = None
    generator_version: Optional[str] = None

    @property
    def max_steps(self) -> int:
        return 4 * self.width * self.height + 20

    def in_bounds(self, c: Coord) -> bool:
        return 0 <= c[0] < self.width and 0 <= c[1] < self.height


@dataclass(frozen=True)
class State:
    segments: Tuple[Coord, ...]
    apples: FrozenSet[Coord]
    step_count: int = 0
    status: Status = Status.PLAYING

    @property
    def head(self) -> Coord:
        return self.segments[0]

    def key(self) -> tuple:
        """Search key used by BFS (step_count excluded, SPEC-03 §2.1)."""
        return (self.segments, self.apples)


@dataclass(frozen=True)
class StepEvent:
    valid_action: bool
    ate_apple: bool
    won: bool
    deadlock: bool
    timeout: bool

    @property
    def invalid_action(self) -> bool:
        return not self.valid_action


@dataclass(frozen=True)
class StepResult:
    state: State
    event: StepEvent
    terminated: bool
    truncated: bool


# ---------------------------------------------------------------- validation

def _coord(v, what: str) -> Coord:
    if (not isinstance(v, (list, tuple)) or len(v) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in v)):
        raise GameError("INVALID_LEVEL", f"{what} must be an integer pair [x,y], got {v!r}")
    return (v[0], v[1])


def make_level(width: int, height: int, walls: Sequence, segments: Sequence,
               apples: Sequence, exit: Sequence, *, ruleset_id: str = RULESET_ID,
               level_id: Optional[str] = None, seed: Optional[int] = None,
               generator_version: Optional[str] = None) -> Level:
    """Build and strictly validate a level; never silently repairs input."""
    if ruleset_id != RULESET_ID:
        raise GameError("UNSUPPORTED_RULESET", f"unsupported ruleset {ruleset_id!r}")
    for name, v in (("width", width), ("height", height)):
        if not isinstance(v, int) or isinstance(v, bool):
            raise GameError("INVALID_LEVEL", f"{name} must be an integer")
    if width != height:
        raise GameError("INVALID_LEVEL", "board must be square")
    if not MIN_SIZE <= width <= MAX_SIZE:
        raise GameError("INVALID_LEVEL", f"board size must be in [{MIN_SIZE},{MAX_SIZE}]")

    wall_list = [_coord(w, "wall") for w in walls]
    seg_list = [_coord(s, "segment") for s in segments]
    apple_list = [_coord(a, "apple") for a in apples]
    ex = _coord(exit, "exit")

    def inb(c):
        return 0 <= c[0] < width and 0 <= c[1] < height

    for c in wall_list + seg_list + apple_list + [ex]:
        if not inb(c):
            raise GameError("INVALID_LEVEL", f"coordinate {list(c)} out of bounds")
    if len(set(wall_list)) != len(wall_list):
        raise GameError("INVALID_LEVEL", "duplicate wall")
    if not MIN_SEGMENTS <= len(seg_list) <= MAX_SEGMENTS:
        raise GameError("INVALID_LEVEL", f"worm must have {MIN_SEGMENTS}-{MAX_SEGMENTS} segments")
    if len(set(seg_list)) != len(seg_list):
        raise GameError("INVALID_LEVEL", "worm segments overlap")
    for a, b in zip(seg_list, seg_list[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            raise GameError("INVALID_LEVEL", "worm segments must be 4-neighbour contiguous")
    if not MIN_APPLES <= len(apple_list) <= MAX_APPLES:
        raise GameError("INVALID_LEVEL", f"level must have {MIN_APPLES}-{MAX_APPLES} apples")
    if len(set(apple_list)) != len(apple_list):
        raise GameError("INVALID_LEVEL", "duplicate apple")

    wset, sset, aset = set(wall_list), set(seg_list), set(apple_list)
    if wset & sset:
        raise GameError("INVALID_LEVEL", "worm overlaps wall")
    if wset & aset:
        raise GameError("INVALID_LEVEL", "apple overlaps wall")
    if sset & aset:
        raise GameError("INVALID_LEVEL", "apple overlaps worm")
    if ex in wset or ex in sset or ex in aset:
        raise GameError("INVALID_LEVEL", "exit overlaps wall/worm/apple")

    level = Level(width, height, frozenset(wset), tuple(seg_list), frozenset(aset), ex,
                  ruleset_id=ruleset_id, level_id=level_id, seed=seed,
                  generator_version=generator_version)
    if not any(valid_action_mask(level, initial_state(level))):
        raise GameError("INVALID_LEVEL", "no legal action in initial state")
    return level


def level_from_dict(d: dict) -> Level:
    """Parse the public LevelDefinition JSON shape (SPEC-05 §2)."""
    try:
        if d.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise GameError("UNSUPPORTED_RULESET", f"unsupported schema_version {d.get('schema_version')!r}")
        return make_level(
            d["grid"]["width"], d["grid"]["height"], d.get("walls", []),
            d["worm"]["segments"], d["apples"], d["exit"],
            ruleset_id=d.get("ruleset_id", RULESET_ID), level_id=d.get("level_id"),
            seed=d.get("seed"), generator_version=d.get("generator_version"))
    except (KeyError, TypeError) as e:
        raise GameError("INVALID_LEVEL", f"malformed level: {e}") from e


def level_to_dict(level: Level) -> dict:
    """Deterministic LevelDefinition serialization (walls/apples sorted)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "ruleset_id": level.ruleset_id,
        "level_id": level.level_id,
        "grid": {"width": level.width, "height": level.height},
        "walls": [list(c) for c in sorted(level.walls)],
        "worm": {"segments": [list(c) for c in level.segments]},
        "apples": [list(c) for c in sorted(level.apples)],
        "exit": list(level.exit),
        "seed": level.seed,
        "generator_version": level.generator_version,
    }


# ---------------------------------------------------------------- transition

def initial_state(level: Level) -> State:
    return State(level.segments, level.apples, 0, Status.PLAYING)


def is_legal(level: Level, segments: Tuple[Coord, ...], apples: FrozenSet[Coord], action: int) -> bool:
    """Single legality function shared by step() and the action mask (SPEC-01 §3.1)."""
    dx, dy = DELTAS[action]
    hx, hy = segments[0]
    nh = (hx + dx, hy + dy)
    if not level.in_bounds(nh):                          # 1. bounds
        return False
    if nh in level.walls:                                # 2. wall
        return False
    if len(segments) >= 2 and nh == segments[1]:         # 3. 180-degree reversal
        return False
    eating = nh in apples
    body = segments if eating else segments[:-1]         # 4. tail leaves only when not eating
    return nh not in body


def advance(segments: Tuple[Coord, ...], apples: FrozenSet[Coord], action: int, exit: Coord):
    """Geometry update for a move already known to be legal (SPEC-01 §3.2 items 2-4).

    Returns ``(segments', apples', ate_apple, won)``. Shared by ``step`` and the solver.
    """
    dx, dy = DELTAS[action]
    nh = (segments[0][0] + dx, segments[0][1] + dy)
    if nh in apples:
        segs = (nh,) + segments
        apples = apples - {nh}
        ate = True
    else:
        segs = (nh,) + segments[:-1]
        ate = False
    return segs, apples, ate, (nh == exit and not apples)


def successors(level: Level, segments: Tuple[Coord, ...], apples: FrozenSet[Coord]):
    """Yield ``(action, segments', apples', won)`` for each legal action in ID order."""
    for a in range(4):
        if is_legal(level, segments, apples, a):
            s2, ap2, _, won = advance(segments, apples, a, level.exit)
            yield a, s2, ap2, won


def _mask(level: Level, segments, apples) -> Tuple[bool, bool, bool, bool]:
    return tuple(is_legal(level, segments, apples, a) for a in range(4))  # type: ignore[return-value]


def valid_action_mask(level: Level, state: State) -> Tuple[bool, bool, bool, bool]:
    return _mask(level, state.segments, state.apples)


def _action_id(action) -> int:
    if isinstance(action, str):
        if action not in ACTION_IDS:
            raise GameError("INVALID_ACTION", f"unknown action {action!r}")
        return ACTION_IDS[action]
    if isinstance(action, bool) or not isinstance(action, int) or not 0 <= int(action) < 4:
        try:
            import numpy as np  # numpy ints from policies
            if isinstance(action, np.integer) and 0 <= int(action) < 4:
                return int(action)
        except ImportError:  # pragma: no cover
            pass
        raise GameError("INVALID_ACTION", f"unknown action {action!r}")
    return int(action)


def step(level: Level, state: State, action) -> StepResult:
    """Pure transition function. Raises EPISODE_FINISHED after terminal states."""
    if state.status != Status.PLAYING:
        raise GameError("EPISODE_FINISHED", f"episode already {state.status.value}; reset first")
    a = _action_id(action)
    steps = state.step_count + 1

    if not is_legal(level, state.segments, state.apples, a):
        timeout = steps >= level.max_steps
        st = State(state.segments, state.apples, steps, Status.TIMEOUT if timeout else Status.PLAYING)
        return StepResult(st, StepEvent(False, False, False, False, timeout), False, timeout)

    segs, apples, ate, won = advance(state.segments, state.apples, a, level.exit)

    if won:                                              # WON has priority over TIMEOUT
        return StepResult(State(segs, apples, steps, Status.WON),
                          StepEvent(True, ate, True, False, False), True, False)
    if not any(_mask(level, segs, apples)):
        return StepResult(State(segs, apples, steps, Status.LOST),
                          StepEvent(True, ate, False, True, False), True, False)
    if steps >= level.max_steps:
        return StepResult(State(segs, apples, steps, Status.TIMEOUT),
                          StepEvent(True, ate, False, False, True), False, True)
    return StepResult(State(segs, apples, steps, Status.PLAYING),
                      StepEvent(True, ate, False, False, False), False, False)


def replay(level: Level, actions: Sequence) -> State:
    """Replay an action list from reset; stops early on terminal states.

    Raises EPISODE_FINISHED if actions remain after a terminal state.
    """
    st = initial_state(level)
    for a in actions:
        st = step(level, st, a).state
    return st


def state_hash(state: State) -> str:
    import hashlib
    import json
    payload = json.dumps({"s": [list(c) for c in state.segments],
                          "a": sorted(list(c) for c in state.apples),
                          "n": state.step_count, "st": state.status.value},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
