"""T-RL-01, T-RL-06, AC-RL-02/03 (SPEC-02 §8)."""
import random

import numpy as np
import pytest

from rl_env import OBS_SHAPE, WormEnv, encode, reward_of
from tests.helpers import demo_level, random_level
from worm_core import Status, initial_state, step, valid_action_mask


@pytest.fixture
def levels():
    rng = random.Random(0)
    out = [demo_level()]
    for i in range(20):
        L = random_level(rng)
        out.append(L.__class__(**{**L.__dict__, "level_id": f"r{i}"}))
    return out


def test_rl01_check_env(levels):
    from gymnasium.utils.env_checker import check_env
    check_env(WormEnv(levels), skip_render_check=True)


def test_rl01_reset_shapes(levels):
    env = WormEnv(levels)
    for s in range(100):
        obs, info = env.reset(seed=s)
        assert obs.shape == OBS_SHAPE and obs.dtype == np.float32
        assert obs.min() >= 0.0 and obs.max() <= 1.0
        assert len(info["action_mask"]) == 4 and any(info["action_mask"])


def test_encoding_channels():
    L = demo_level()
    o = encode(L, initial_state(L))
    assert o[1, 2, 1] == 1.0 and o[1, 3, 1] == pytest.approx(0.9) and o[1, 4, 1] == pytest.approx(0.8)
    assert o[2, 2, 3] == 1.0 and o[3, 2, 4] == 1.0
    assert o[4].sum() == 25 and o[4, 5:, :].sum() == 0 and o[5, 0, 0] == 1.0 and o[5, 7, 7] == 0.0


def test_fixture_episode_rewards():
    env = WormEnv([demo_level()])
    env.reset(seed=0, options={"level_id": "demo-5x5-001"})
    rs = [env.step(1) for _ in range(3)]
    assert [round(r[1], 4) for r in rs] == [-0.01, 0.99, 9.99]
    assert rs[-1][2] and rs[-1][4]["won"]


def test_ac_rl02_mask_matches_core(levels):
    env = WormEnv(levels)
    rng = random.Random(1)
    checked = 0
    for ep in range(400):
        env.reset(seed=ep)
        done = False
        while not done:
            assert tuple(env.action_masks()) == valid_action_mask(env.level, env.state)
            checked += 1
            _, _, term, trunc, _ = env.step(rng.randrange(4))
            done = term or trunc
    assert checked >= 10_000


def test_ac_rl03_deterministic(levels):
    def roll(seed):
        env = WormEnv(levels)
        env.reset(seed=seed)
        rng, out, done = random.Random(seed), [], False
        while not done:
            o, r, term, trunc, _ = env.step(rng.randrange(4))
            out.append((r, o.tobytes()))
            done = term or trunc
        return out
    assert roll(3) == roll(3)


def test_rl06_terminal_states_do_not_crash_masked_sampling():
    from worm_core import make_level
    L = make_level(5, 5, [(0, 1)], [(1, 0), (2, 0)], [(4, 4)], (3, 3), level_id="dead")
    env = WormEnv([L])
    env.reset(seed=0)
    _, _, term, trunc, info = env.step(3)
    assert term and info["deadlock"] and not any(env.action_masks())


def test_reward_matches_spec():
    L = demo_level()
    ev = step(L, initial_state(L), "DOWN").event
    assert round(reward_of(ev), 4) == -0.06
