"""T-CORE-06/07/08: sampled-state invariants (seeded, deterministic)."""
import random

from tests.helpers import random_level
from worm_core import DELTAS, Status, initial_state, replay, state_hash, step, valid_action_mask


def reference_legal(L, segs, apples, a):
    """Independent re-implementation of SPEC-01 §3.1 for differential testing."""
    hx, hy = segs[0]
    nx, ny = hx + DELTAS[a][0], hy + DELTAS[a][1]
    if nx < 0 or ny < 0 or nx >= L.width or ny >= L.height:
        return False
    if (nx, ny) in L.walls:
        return False
    if (nx, ny) == segs[1]:
        return False
    occupied = list(segs) if (nx, ny) in apples else list(segs)[:-1]
    return (nx, ny) not in occupied


def sampled_states(n_states, seed=1234):
    rng = random.Random(seed)
    count = 0
    while count < n_states:
        L = random_level(rng)
        s = initial_state(L)
        while s.status == Status.PLAYING and count < n_states:
            yield L, s
            count += 1
            s = step(L, s, rng.randrange(4)).state


def test_core06_07_invariants_and_mask():
    n = 0
    for L, s in sampled_states(12_000):
        segs = s.segments
        assert len(set(segs)) == len(segs)
        assert all(L.in_bounds(c) and c not in L.walls for c in segs)
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(segs, segs[1:]))
        assert not (set(segs) & s.apples)
        mask = valid_action_mask(L, s)
        assert any(mask), "PLAYING state must never have an all-false mask"
        for a in range(4):
            assert mask[a] == reference_legal(L, segs, s.apples, a)
            r = step(L, s, a)
            assert r.event.valid_action == mask[a]
            if not mask[a]:
                assert r.state.segments == segs and r.state.step_count == s.step_count + 1
        n += 1
    assert n >= 10_000


def test_core08_replay_determinism():
    rng = random.Random(7)
    for _ in range(20):
        L = random_level(rng)
        actions, s = [], initial_state(L)
        while s.status == Status.PLAYING:
            a = rng.randrange(4)
            actions.append(a)
            s = step(L, s, a).state
        hashes = {state_hash(replay(L, actions)) for _ in range(100)}
        assert len(hashes) == 1
