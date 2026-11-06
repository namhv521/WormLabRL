"""Gymnasium adapter over ``worm_core`` (SPEC-02 §2). Defines no rules of its own."""
from __future__ import annotations

from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from worm_core import ACTIONS, RULESET_ID, Level, State, initial_state, step, valid_action_mask

OBS_SHAPE = (6, 8, 8)
PAD = 8
ACTION_MAPPING = list(ACTIONS)

R_STEP, R_APPLE, R_WIN, R_INVALID = -0.01, 1.0, 10.0, -0.05


def reward_of(event) -> float:
    """Shared reward (SPEC-02 §4); the session API reuses it for debugging."""
    return (R_STEP + (R_APPLE if event.ate_apple else 0.0) + (R_WIN if event.won else 0.0)
            + (R_INVALID if event.invalid_action else 0.0))


def encode(level: Level, state: State) -> np.ndarray:
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    n = level.width
    for x, y in level.walls:
        obs[0, y, x] = 1.0
    for i, (x, y) in enumerate(state.segments):
        obs[1, y, x] = 1.0 - i / 10.0
    for x, y in state.apples:
        obs[2, y, x] = 1.0
    obs[3, level.exit[1], level.exit[0]] = 1.0
    obs[4, :n, :n] = 1.0
    obs[5, :n, :n] = (level.max_steps - state.step_count) / level.max_steps
    return obs


class WormEnv(gym.Env):
    """Samples a level from ``levels`` on each reset (or ``options['level_id']``).

    ``weights`` optionally biases sampling (curriculum); it can be changed
    between episodes with :meth:`set_weights`.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, levels: Sequence[Level], weights: Optional[Sequence[float]] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        if not levels:
            raise ValueError("WormEnv needs at least one level")
        if any(L.ruleset_id != RULESET_ID for L in levels):
            raise ValueError("all levels must use ruleset " + RULESET_ID)
        self.levels = list(levels)
        self.by_id = {L.level_id: L for L in self.levels if L.level_id}
        self.set_weights(weights)
        self.observation_space = spaces.Box(0.0, 1.0, OBS_SHAPE, dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode
        self.level: Optional[Level] = None
        self.state: Optional[State] = None

    def set_weights(self, weights):
        if weights is None:
            self.p = None
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (len(self.levels),) or w.sum() <= 0:
                raise ValueError("weights must be non-negative, one per level, not all zero")
            self.p = w / w.sum()

    def _info(self, event=None) -> dict:
        info = {"level_id": self.level.level_id, "step_count": self.state.step_count,
                "action_mask": list(valid_action_mask(self.level, self.state))}
        if event is not None:
            info.update(ate_apple=event.ate_apple, won=event.won, invalid_action=event.invalid_action,
                        deadlock=event.deadlock, timeout=event.timeout)
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        lid = (options or {}).get("level_id")
        if lid is not None:
            if lid not in self.by_id:
                raise KeyError(f"unknown level_id {lid!r}")
            self.level = self.by_id[lid]
        else:
            self.level = self.levels[int(self.np_random.choice(len(self.levels), p=self.p))]
        self.state = initial_state(self.level)
        return encode(self.level, self.state), self._info()

    def step(self, action):
        r = step(self.level, self.state, int(action))
        self.state = r.state
        return encode(self.level, self.state), reward_of(r.event), r.terminated, r.truncated, self._info(r.event)

    def action_masks(self) -> np.ndarray:
        return np.array(valid_action_mask(self.level, self.state), dtype=bool)

    def render(self):
        L, s = self.level, self.state
        grid = [["." for _ in range(L.width)] for _ in range(L.height)]
        for x, y in L.walls:
            grid[y][x] = "#"
        grid[L.exit[1]][L.exit[0]] = "E"
        for x, y in s.apples:
            grid[y][x] = "a"
        for i, (x, y) in enumerate(s.segments):
            grid[y][x] = "H" if i == 0 else "o"
        return "\n".join("".join(r) for r in grid) + f"\n{s.status.value} step {s.step_count}/{L.max_steps}"
