"""Core physics adapter, rich spatial + 1-step transition encoder, and dataset utilities."""

from __future__ import annotations

from collections import Counter, deque
from copy import copy
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from worm_physics import DIRECTIONS, HAZARDS, Rules, State, WormMap, load_maps, trace_hints

W: int = 22
H: int = 12
MAX_STEPS: int = 250
GRID_CHANNELS: int = 12
ACTION_FEATURES: int = 26

ACTIONS: tuple[int, ...] = (1, 2, 3, 4)
ACTION_DELTAS: dict[int, tuple[int, int]] = dict(DIRECTIONS)
ACTION_NAMES: dict[int, str] = {1: "LEFT", 2: "RIGHT", 3: "UP", 4: "DOWN"}
BLOCKED_EVENTS: frozenset[str] = frozenset({"wall", "body", "stone"})
FATAL_EVENTS: frozenset[str] = frozenset({"trap", "fall", "crushed", "deadlock", "stuck"})
STRICT_RULES = Rules(
    apple_support=True,
    portal_support=True,
    hazard_support=False,
    worm_supports_stone=True,
    push_vertical=True,
    portal_needs_apples=False,
    stone_on_hazard=True,
    chain_push=True,
    stone_stops_on_hazard=True,
    destroy_stone_on_hazard=False,
)


def has_surviving_move(board: WormMap, state: State) -> bool:
    """Return True if at least one action from `state` leads to 'ok' or 'win'."""
    for act in ACTIONS:
        _, ev = board.step(state, act)
        if ev in ("ok", "win"):
            return True
    return False


def step_strict(board: WormMap, state: State, action: int) -> tuple[State, str]:
    """Execute one physics step with immediate deadlock detection when 0 surviving moves remain."""
    after, event = board.step(state, action)
    if event == "ok" and not has_surviving_move(board, after):
        return after, "deadlock"
    return after, event


def step_strict_trace(board: WormMap, state: State, action: int) -> tuple[list[State], str]:
    """Execute one physics step returning all intermediate gravity-drop frames + deadlock check."""
    sub_states, event = board.step_trace(state, action)
    if event == "ok" and not has_surviving_move(board, sub_states[-1]):
        return sub_states, "deadlock"
    return sub_states, event

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEVELS_JSON = ROOT / "hungry_worm_levels.json"
DEFAULT_VIEWER_HTML = ROOT / "hungry_worm_map_viewer.html"


def auto_orient_hazard(x: int, y: int, walls: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> int:
    """Choose authentic thorn orientation attached to adjacent wall, or spinning gear if floating."""
    if (x, y - 1) in walls:
        return 3   # Sitting on top of floor, pointing UP
    if (x, y + 1) in walls:
        return 10  # Hanging from ceiling, pointing DOWN
    if (x - 1, y) in walls:
        return 11  # Attached to left wall, pointing RIGHT
    if (x + 1, y) in walls:
        return 9   # Attached to right wall, pointing LEFT
    return 7       # Floating gear


def bricks_from_components(
    walls: Iterable[tuple[int, int]],
    stones: Iterable[tuple[int, int]],
    apples: Iterable[tuple[int, int]],
    hazards: dict[tuple[int, int], int] | Iterable[tuple[int, int]],
    portal: tuple[int, int],
    worm_body: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> list[dict[str, int]]:
    """Serialize level components into standard Hungry Worm bricks JSON format."""
    wall_set = set(walls)
    bricks: list[dict[str, int]] = []
    for x, y in sorted(wall_set):
        bricks.append({"x": int(x), "y": int(y), "id": 1})
    for x, y in sorted(set(stones)):
        bricks.append({"x": int(x), "y": int(y), "id": 2})
    for x, y in sorted(set(apples)):
        bricks.append({"x": int(x), "y": int(y), "id": 4})
    if isinstance(hazards, dict):
        for (x, y), hid in sorted(hazards.items()):
            bricks.append({"x": int(x), "y": int(y), "id": int(hid)})
    else:
        for x, y in sorted(set(hazards)):
            bricks.append({"x": int(x), "y": int(y), "id": auto_orient_hazard(x, y, wall_set)})
    bricks.append({"x": int(portal[0]), "y": int(portal[1]), "id": 8})
    hx, hy = worm_body[0]
    bricks.append({"x": int(hx), "y": int(hy), "id": 5})
    for bx, by in worm_body[1:]:
        bricks.append({"x": int(bx), "y": int(by), "id": 6})
    return bricks


def validate_initial_level(level: dict, rules: Rules = STRICT_RULES) -> tuple[bool, str, WormMap | None]:
    """Verify a level dict is structurally valid, unambiguous, settled at frame 0, and non-deadlocked."""
    if level.get("width", W) != W or level.get("height", H) != H:
        return False, "invalid_dimensions", None
    bricks = level.get("bricks", [])
    coords = [(b["x"], b["y"]) for b in bricks]
    if len(coords) != len(set(coords)):
        return False, "overlapping_bricks", None
    if any(not (0 <= x < W and 0 <= y < H) for x, y in coords):
        return False, "out_of_bounds_brick", None
    heads = [b for b in bricks if b["id"] == 5]
    bodies = [b for b in bricks if b["id"] == 6]
    portals = [b for b in bricks if b["id"] == 8]
    if len(heads) != 1 or len(portals) != 1 or len(bodies) < 1:
        return False, "missing_worm_or_portal", None
    try:
        board = WormMap(level, rules=rules)
    except Exception as exc:
        return False, f"worm_init_error:{exc}", None
    settled, event = board.settle(board.initial)
    if event != "ok" or settled != board.initial:
        return False, f"unstable_frame0:{event}", None
    if not has_surviving_move(board, board.initial):
        return False, "frame0_deadlock", None
    return True, "ok", board


def geometry_signature(level: dict) -> tuple:
    """Canonical signature invariant to brick order and horizontal mirror reflection."""
    cells = sorted((int(b["x"]), int(b["y"]), int(b["id"])) for b in level["bricks"])
    mirror_id = {9: 11, 11: 9}
    width = int(level.get("width", W))
    mirrored = sorted((width - 1 - x, y, mirror_id.get(kind, kind)) for x, y, kind in cells)
    return (width, int(level.get("height", H)), tuple(min(cells, mirrored)))


def load_levels_json(path: Path | str = DEFAULT_LEVELS_JSON) -> list[tuple[str, dict]]:
    """Load (pack, level_dict) pairs from JSON or viewer HTML."""
    path = Path(path)
    if path.suffix.lower() == ".html":
        return load_maps(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "levelsA" in data:
        rows = [(pack, lv) for pack in ("A", "B") for lv in data.get("levels" + pack, [])]
        for extra_key in ("levelsG", "generated", "levels"):
            if extra_key in data and isinstance(data[extra_key], list):
                rows.extend(("G", lv) for lv in data[extra_key])
        return rows
    if isinstance(data, dict) and "bricks" in data:
        return [("C", data)]
    if isinstance(data, list):
        return [("C", lv) for lv in data]
    raise ValueError(f"Unrecognized level file structure in {path}")


class LevelArena:
    """Fast level wrapper providing cached transitions, BFS distance fields, and dual-stream observations."""

    def __init__(self, board: WormMap, name: str = "custom", level_dict: dict | None = None):
        self.board = board
        self.name = name
        self.level_dict = level_dict
        self.goal = min(board.portals)
        self.distance: dict[tuple[int, int], int] = {p: 0 for p in board.portals}
        queue = deque(sorted(board.portals))
        blocked_static = board.walls | board.hazards
        while queue:
            x, y = queue.popleft()
            d = self.distance[(x, y)]
            for dx, dy in ACTION_DELTAS.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < board.width and 0 <= ny < board.height:
                    np_pt = (nx, ny)
                    if np_pt not in self.distance and np_pt not in blocked_static:
                        self.distance[np_pt] = d + 1
                        queue.append(np_pt)

        # Precompute static spatial channels (12, H, W)
        self.base_grid = np.zeros((GRID_CHANNELS, H, W), dtype=np.float32)
        for x, y in board.walls:
            if 0 <= x < W and 0 <= y < H:
                self.base_grid[0, y, x] = 1.0
        for x, y in board.hazards:
            if 0 <= x < W and 0 <= y < H:
                self.base_grid[3, y, x] = 1.0
        for x, y in board.portals:
            if 0 <= x < W and 0 <= y < H:
                self.base_grid[4, y, x] = 1.0
        # Ch 9: normalized static topological distance to portal (1.0 near portal, 0.0 far/unreachable)
        for y in range(H):
            for x in range(W):
                if (x, y) in self.distance:
                    self.base_grid[9, y, x] = max(0.0, 1.0 - self.distance[(x, y)] / 40.0)
        self.base_grid[10, :, :] = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        self.base_grid[11, :, :] = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None]

        self._step_all = lru_cache(maxsize=65536)(self._compute_transitions)

    def dist(self, point: tuple[int, int]) -> int:
        if point in self.distance:
            return self.distance[point]
        return 50 + abs(point[0] - self.goal[0]) + abs(point[1] - self.goal[1])

    def admissible_heuristic(self, head: tuple[int, int]) -> int:
        """100% admissible lower bound on actions needed to reach a portal under gravity."""
        hx, hy = head
        return min(abs(hx - gx) + max(0, gy - hy) for gx, gy in self.board.portals)

    def _compute_transitions(self, state: State):
        """Compute 1-step transitions and state-only action features (excluding visited/steps_left)."""
        board = self.board
        x, y = state.body[0]
        gx, gy = self.goal
        cur_dist = self.dist((x, y))
        occupied = set(state.body)
        supports = board.support | state.stones
        if board.rules.apple_support:
            supports = supports | state.apples

        cur_apple_dist = min((abs(x - ax) + abs(y - ay) for ax, ay in state.apples), default=0)
        transitions: list[tuple[State, str]] = []
        base_feats = np.zeros((4, ACTION_FEATURES), dtype=np.float32)

        for idx, action in enumerate(ACTIONS):
            dx, dy = ACTION_DELTAS[action]
            target = (x + dx, y + dy)
            after, event = step_strict(board, state, action)
            transitions.append((after, event))
            nx, ny = after.body[0]
            after_dist = self.dist((nx, ny))
            after_supports = board.support | after.stones
            if board.rules.apple_support:
                after_supports = after_supports | after.apples
            supported_ratio = sum((bx, by - 1) in after_supports for bx, by in after.body) / len(after.body)
            next_apple_dist = min((abs(nx - ax) + abs(ny - ay) for ax, ay in after.apples), default=0)

            # Count post-move non-blocked actions (1-step mobility / deadlock awareness)
            mobility = 0
            if event == "ok":
                after_occ = set(after.body[1:-1]) | set(after.body[1:2])
                for ndx, ndy in ACTION_DELTAS.values():
                    nt = (nx + ndx, ny + ndy)
                    if nt not in board.walls and nt not in after_occ:
                        mobility += 1

            base_feats[idx] = (
                dx,
                dy,
                (gx - x) / W,
                (gy - y) / H,
                (gx - x) * dx / W,
                (gy - y) * dy / H,
                max(-1.0, min(1.0, (cur_dist - after_dist) / 4.0)),
                min(cur_dist / 40.0, 2.0),
                max(-1.0, min(1.0, (ny - y) / 6.0)),
                len(state.body) / 16.0,
                float(event in BLOCKED_EVENTS),          # idx 10
                float(event in FATAL_EVENTS),            # idx 11
                float(event == "win"),                   # idx 12
                float(len(after.apples) < len(state.apples)),  # idx 13: ate apple
                float(after.stones != state.stones),     # idx 14: pushed/dropped stone
                float(target in board.walls),
                float(target in board.hazards),
                float(target in occupied),
                float(target in state.stones),
                float(target in state.apples),
                supported_ratio,
                next_apple_dist / W,
                0.0,  # idx 22: reserved for `after in seen` (filled dynamically)
                max(-1.0, min(1.0, (cur_apple_dist - next_apple_dist) / 4.0)) if state.apples else 0.0,
                len(after.apples) / 8.0,
                mobility / 4.0,
            )
        return base_feats, tuple(transitions)

    def encode_grid(self, state: State) -> np.ndarray:
        """Encode current board & worm topology into (12, H, W) float32 spatial tensor."""
        grid = self.base_grid.copy()
        for x, y in state.stones:
            if 0 <= x < W and 0 <= y < H:
                grid[1, y, x] = 1.0
        for x, y in state.apples:
            if 0 <= x < W and 0 <= y < H:
                grid[2, y, x] = 1.0
        length = len(state.body)
        for idx, (x, y) in enumerate(state.body):
            if 0 <= x < W and 0 <= y < H:
                grid[7, y, x] = (length - idx) / length
        hx, hy = state.body[0]
        tx, ty = state.body[-1]
        if 0 <= hx < W and 0 <= hy < H:
            grid[5, hy, hx] = 1.0
        if 0 <= tx < W and 0 <= ty < H:
            grid[6, ty, tx] = 1.0
        supports = self.board.support | state.stones
        if self.board.rules.apple_support:
            supports = supports | state.apples
        for x, y in state.body:
            if 0 <= x < W and 0 <= y < H and (x, y - 1) in supports:
                grid[8, y, x] = 1.0
        return grid

    def observe(
        self,
        state: State,
        seen: set[State] | frozenset[State] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[tuple[State, str], ...]]:
        """Return (grid, action_feats, safe_mask, legal_mask, transitions)."""
        base_feats, transitions = self._step_all(state)
        feats = base_feats.copy()
        if seen is not None:
            for i, (after, _) in enumerate(transitions):
                if after in seen:
                    feats[i, 22] = 1.0

        legal = ~(feats[:, 10].astype(bool))
        safe = legal & ~(feats[:, 11].astype(bool))
        # Lossless safe mask: never prunes any winning path because fatal moves terminate in death
        effective_safe = safe if safe.any() else (legal if legal.any() else np.ones(4, dtype=bool))
        effective_legal = legal if legal.any() else np.ones(4, dtype=bool)
        grid = self.encode_grid(state)
        return grid, feats, effective_safe, effective_legal, transitions

    def to_level_dict(self, level_id: int = 1, hints: list[dict] | None = None) -> dict:
        """Export this arena back to standard Hungry Worm level dict."""
        if self.level_dict is not None and hints is None:
            return self.level_dict
        bricks = bricks_from_components(
            walls=self.board.walls,
            stones=self.board.initial.stones,
            apples=self.board.initial.apples,
            hazards=self.board.hazards,
            portal=self.goal,
            worm_body=self.board.initial.body,
        )
        return {
            "id": int(level_id),
            "width": self.board.width,
            "height": self.board.height,
            "bricks": bricks,
            "hints": hints if hints is not None else (self.level_dict.get("hints", []) if self.level_dict else []),
        }


def mirror_arena_and_trace(
    arena: LevelArena,
    trace: list[tuple[State, int, State, str]],
) -> tuple[LevelArena, list[tuple[State, int, State, str]]] | None:
    """Horizontally mirror a level and its verified solution trajectory for 2x data augmentation."""
    width = arena.board.width

    def flip_pt(pt: tuple[int, int]) -> tuple[int, int]:
        return (width - 1 - pt[0], pt[1])

    def flip_set(pts: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
        return frozenset(flip_pt(p) for p in pts)

    def flip_state(st: State) -> State:
        return State(tuple(flip_pt(p) for p in st.body), flip_set(st.apples), flip_set(st.stones))

    board = copy(arena.board)
    for attr in ("walls", "hazards", "portals", "support", "fixed"):
        setattr(board, attr, flip_set(getattr(board, attr)))
    board.initial = flip_state(board.initial)
    action_flip = {1: 2, 2: 1, 3: 3, 4: 4}
    mirrored_trace = [
        (flip_state(s), action_flip[a], flip_state(ns), ev)
        for s, a, ns, ev in trace
    ]
    # Verify exact replay on mirrored board
    cur = board.initial
    for idx, (s, a, ns, ev) in enumerate(mirrored_trace):
        if s != cur:
            return None
        actual_ns, actual_ev = board.step(cur, a)
        if actual_ns != ns or actual_ev != ev:
            return None
        if ev != ("win" if idx == len(mirrored_trace) - 1 else "ok"):
            return None
        cur = actual_ns
    return LevelArena(board, name=f"{arena.name}_mirror"), mirrored_trace


def load_dataset(path: Path | str = DEFAULT_LEVELS_JSON) -> dict[str, list[tuple[dict, LevelArena, list]]]:
    """Load and partition the 240 canonical levels into train (141), val (23), test (40) splits."""
    raw_levels = load_levels_json(path)
    groups: dict[tuple, str] = {}
    unique_count = 0
    splits: dict[str, list[tuple[dict, LevelArena, list]]] = {
        "train": [],
        "val": [],
        "test": [],
        "duplicate": [],
        "unverified": [],
    }
    train_pool: list[tuple[dict, LevelArena, list]] = []

    for pack, level in raw_levels:
        name = f"{pack}{level['id']}"
        board = WormMap(level, rules=STRICT_RULES)
        settled_init, init_ev = board.settle(board.initial)
        if init_ev != "ok" or settled_init != board.initial:
            trace, reason = [], f"unstable_init:{init_ev}"
        else:
            trace, reason = trace_hints(board, level.get("hints", []))
        meta = {
            "name": name,
            "pack": pack,
            "id": level["id"],
            "width": board.width,
            "height": board.height,
            "hint_steps": len(level.get("hints", [])),
            "verified_steps": len(trace),
            "verification": reason,
            "level": level,
        }
        arena = LevelArena(board, name=name, level_dict=level)
        if reason != "win":
            meta["split"] = "unverified"
            splits["unverified"].append((meta, arena, trace))
            continue
        sig = geometry_signature(level)
        if sig in groups:
            meta["split"] = "duplicate"
            meta["alias"] = groups[sig]
            splits["duplicate"].append((meta, arena, trace))
            continue
        groups[sig] = name
        unique_count += 1
        if unique_count % 5 == 0:
            meta["split"] = "test"
            splits["test"].append((meta, arena, trace))
        else:
            train_pool.append((meta, arena, trace))

    for idx, item in enumerate(train_pool, 1):
        if idx % 7 == 0:
            item[0]["split"] = "val"
            splits["val"].append(item)
        else:
            item[0]["split"] = "train"
            splits["train"].append(item)

    return splits
