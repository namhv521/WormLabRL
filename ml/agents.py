"""Agents and the shared episode runner used by evaluation, baselines and the API.

An agent only sees ``(level, state)`` and returns an action ID. ``RLAgent``
never imports the solver: inference is the policy's own decisions (SPEC-02 §1).
"""
from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np

from worm_core import ACTIONS, DELTAS, Level, State, Status, initial_state, step, valid_action_mask


class Agent:
    name = "agent"

    def reset(self, level: Level) -> None:
        pass

    def act(self, level: Level, state: State) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    """Uniform over currently legal actions."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, level, state):
        legal = [a for a, ok in enumerate(valid_action_mask(level, state)) if ok]
        return self.rng.choice(legal)


class GreedyAgent(Agent):
    """One-step Manhattan greedy toward the nearest apple, then the exit; ties by action ID."""

    name = "greedy"

    def act(self, level, state):
        targets = state.apples or {level.exit}
        hx, hy = state.head
        best, best_d = None, None
        for a, ok in enumerate(valid_action_mask(level, state)):
            if not ok:
                continue
            nx, ny = hx + DELTAS[a][0], hy + DELTAS[a][1]
            d = min(abs(nx - tx) + abs(ny - ty) for tx, ty in targets)
            if best_d is None or d < best_d:
                best, best_d = a, d
        return best


class BFSAgent(Agent):
    """Oracle: replays the exact BFS path. Benchmark/oracle use only."""

    name = "bfs"

    def reset(self, level):
        from solver import SOLVED, solve
        res = solve(level, time_budget_ms=None)
        self.plan = [ACTIONS.index(a) for a in res.actions] if res.status == SOLVED else []
        self.i = 0

    def act(self, level, state):
        if self.i >= len(self.plan):
            return 0
        a = self.plan[self.i]
        self.i += 1
        return a


class RLAgent(Agent):
    """Wraps a MaskablePPO model; deterministic masked inference."""

    name = "rl"

    def __init__(self, model, model_id: Optional[str] = None):
        self.model = model
        self.model_id = model_id

    def act(self, level, state):
        from rl_env import encode
        mask = np.array(valid_action_mask(level, state), dtype=bool)
        a, _ = self.model.predict(encode(level, state), action_masks=mask, deterministic=True)
        return int(a)


def outcome_of(status: Status) -> str:
    return {Status.WON: "SOLVED", Status.LOST: "DEADLOCK", Status.TIMEOUT: "TIMEOUT"}.get(status, "FAILED")


def run_episode(agent: Agent, level: Level, deadline_s: Optional[float] = None) -> dict:
    """Run from reset to terminal/truncation. ``deadline_s`` bounds wall time (API use)."""
    t0 = time.perf_counter()
    agent.reset(level)
    st = initial_state(level)
    actions = []
    lat = []
    while st.status == Status.PLAYING:
        if deadline_s is not None and time.perf_counter() - t0 > deadline_s:
            return {"outcome": "INFERENCE_TIMEOUT", "actions": actions, "steps": len(actions),
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000)}
        ta = time.perf_counter()
        a = agent.act(level, st)
        lat.append((time.perf_counter() - ta) * 1000)
        actions.append(ACTIONS[a])
        st = step(level, st, a).state
    return {"outcome": outcome_of(st.status), "actions": actions, "steps": st.step_count,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "p95_act_ms": float(np.percentile(lat, 95)) if lat else 0.0}
