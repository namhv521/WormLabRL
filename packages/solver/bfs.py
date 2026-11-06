"""Exact BFS solver + certificate verification (SPEC-03 §2, §5.2)."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from worm_core import ACTIONS, RULESET_ID, RULESET_VERSION, GameError, Level, Status, level_hash, replay
from worm_core.rules import successors

SOLVER_VERSION = "bfs-v1"
VERIFIER_VERSION = RULESET_VERSION
DEFAULT_NODE_BUDGET = 150_000
DEFAULT_TIME_BUDGET_MS = 1_500

SOLVED, EXHAUSTED, LIMIT, INVALID_LEVEL = "SOLVED", "EXHAUSTED", "LIMIT", "INVALID_LEVEL"


@dataclass
class SolveResult:
    status: str
    expanded_nodes: int
    elapsed_ms: int
    actions: Optional[List[str]] = None
    optimal_steps: Optional[int] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def solve(level: Level, node_budget: int = DEFAULT_NODE_BUDGET,
          time_budget_ms: Optional[int] = DEFAULT_TIME_BUDGET_MS) -> SolveResult:
    """Breadth-first search over (segments, remaining_apples).

    * Actions expanded in fixed order UP, RIGHT, DOWN, LEFT -> deterministic result.
    * ``expanded_nodes`` counts every state popped from the FIFO queue (goal and
      deadlock states included), before the goal test.
    * SOLVED only when the path also fits in ``max_steps`` and replays to WON.
    * ``time_budget_ms=None`` disables the wall-clock limit (fully deterministic).
    """
    t0 = time.perf_counter()
    start = (level.segments, level.apples)
    parent = {start: None}            # key -> (prev_key, action)
    goal = set()
    queue = deque([start])
    expanded = 0
    deadline = None if time_budget_ms is None else t0 + time_budget_ms / 1000.0

    def ms():
        return int((time.perf_counter() - t0) * 1000)

    while queue:
        if expanded >= node_budget:
            return SolveResult(LIMIT, expanded, ms(), reason="node budget exhausted")
        if deadline is not None and (expanded & 255) == 0 and time.perf_counter() > deadline:
            return SolveResult(LIMIT, expanded, ms(), reason="time budget exhausted")
        key = queue.popleft()
        expanded += 1
        if key in goal:
            actions = _reconstruct(parent, key)
            res = SolveResult(SOLVED, expanded, ms(), actions=actions, optimal_steps=len(actions))
            if len(actions) > level.max_steps:
                return SolveResult(LIMIT, expanded, ms(),
                                   reason=f"shortest solution {len(actions)} exceeds max_steps {level.max_steps}")
            if not verify_actions(level, actions):
                return SolveResult("INTERNAL_VERIFICATION_FAILED", expanded, ms(),
                                   reason="BFS path failed engine replay")
            return res
        segs, apples = key
        for a, s2, ap2, won in successors(level, segs, apples):
            k2 = (s2, ap2)
            if k2 in parent:
                continue
            parent[k2] = (key, a)
            if won:
                goal.add(k2)
            queue.append(k2)
    return SolveResult(EXHAUSTED, expanded, ms(), reason="reachable state space exhausted without reaching goal")


def _reconstruct(parent, key) -> List[str]:
    out = []
    while parent[key] is not None:
        key, a = parent[key]
        out.append(ACTIONS[a])
    out.reverse()
    return out


def verify_actions(level: Level, actions) -> bool:
    """Independent replay through the core from reset; True only on exact WON at the last action."""
    try:
        st = replay(level, actions)
    except GameError:
        return False
    return st.status == Status.WON and st.step_count == len(actions) <= level.max_steps


def make_certificate(level: Level, result: SolveResult) -> dict:
    if result.status != SOLVED:
        raise GameError("INTERNAL_VERIFICATION_FAILED", f"cannot certify status {result.status}")
    ok = verify_actions(level, result.actions)
    if not ok:
        raise GameError("INTERNAL_VERIFICATION_FAILED", "certificate replay failed")
    return {
        "level_hash": level_hash(level),
        "ruleset_id": RULESET_ID,
        "solver_version": SOLVER_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "actions": list(result.actions),
        "optimal_steps": result.optimal_steps,
        "expanded_nodes": result.expanded_nodes,
        "verified_by_replay": True,
    }


def verify_certificate(level: Level, cert: dict) -> bool:
    """Check a stored certificate against the level actually being served."""
    try:
        return (cert.get("ruleset_id") == level.ruleset_id
                and cert.get("level_hash") == level_hash(level)
                and cert.get("verified_by_replay") is True
                and isinstance(cert.get("actions"), list)
                and cert.get("optimal_steps") == len(cert["actions"])
                and verify_actions(level, cert["actions"]))
    except Exception:
        return False
