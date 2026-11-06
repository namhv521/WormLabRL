"""T-CORE-01..11 deterministic unit tests (SPEC-06 §2)."""
import pytest

from tests.helpers import demo_level
from worm_core import (DELTAS, GameError, Status, initial_state, make_level, replay,
                       state_hash, step, valid_action_mask)
from worm_core.rules import State, is_legal


def lvl(segments, apples=((4, 4),), exit=(3, 3), walls=(), n=5):
    return make_level(n, n, list(walls), list(segments), list(apples), exit)


# ---- golden fixture (SPEC-06 §2.1)
def test_fixture_initial_mask():
    L = demo_level()
    assert valid_action_mask(L, initial_state(L)) == (True, True, False, True)
    assert L.max_steps == 120


def test_fixture_solution_wins():
    L = demo_level()
    st = replay(L, ["RIGHT", "RIGHT", "RIGHT"])
    assert st.status == Status.WON and st.step_count == 3
    assert st.apples == frozenset() and len(st.segments) == 4


def test_fixture_step_details():
    L = demo_level()
    r1 = step(L, initial_state(L), "RIGHT")
    assert r1.state.segments == ((2, 2), (1, 2), (1, 3))
    assert valid_action_mask(L, r1.state) == (True, True, True, False)
    r2 = step(L, r1.state, "RIGHT")
    assert r2.event.ate_apple and len(r2.state.segments) == 4


def test_fixture_down_invalid_up_valid():
    L = demo_level()
    s0 = initial_state(L)
    r = step(L, s0, "DOWN")
    assert not r.event.valid_action and r.state.step_count == 1 and r.state.segments == s0.segments
    assert step(L, s0, "UP").event.valid_action


# ---- T-CORE-01: deltas + determinism
def test_core01_deltas_and_determinism():
    assert DELTAS == ((0, -1), (1, 0), (0, 1), (-1, 0))
    L = lvl([(2, 2), (2, 3)])
    s = initial_state(L)
    for a, exp in ((0, (2, 1)), (1, (3, 2)), (3, (1, 2))):
        r = step(L, s, a)
        assert r.state.head == exp
        assert step(L, s, a) == r


# ---- T-CORE-02: wall / bounds / body -> unchanged geometry, step+1
@pytest.mark.parametrize("segments,walls,action", [
    ([(0, 0), (0, 1)], [], "UP"),                             # bounds
    ([(0, 0), (0, 1)], [], "LEFT"),                           # bounds
    ([(2, 2), (2, 3)], [(3, 2)], "RIGHT"),                    # wall
    ([(2, 2), (2, 3)], [], "DOWN"),                           # 180 reversal (len 2)
    ([(1, 1), (2, 1), (2, 2), (1, 2), (0, 2)], [], "DOWN"),   # middle body
])
def test_core02_blocked(segments, walls, action):
    L = lvl(segments, walls=walls)
    s = initial_state(L)
    r = step(L, s, action)
    assert r.event.invalid_action
    assert r.state.segments == s.segments and r.state.apples == s.apples
    assert r.state.step_count == 1 and r.state.status == Status.PLAYING


def test_core02_tail_chase_allowed():
    L = lvl([(1, 1), (2, 1), (2, 2), (1, 2)])
    r = step(L, initial_state(L), "DOWN")       # into tail cell which is leaving
    assert r.event.valid_action and r.state.segments == ((1, 2), (1, 1), (2, 1), (2, 2))


def test_core02_tail_blocked_when_eating():
    # When eating, the tail stays, so every body cell (including tail) is forbidden.
    L = lvl([(1, 1), (2, 1), (2, 2), (1, 2)])
    assert not is_legal(L, L.segments, frozenset({(1, 2)}), 2)


# ---- T-CORE-03: eat apple
def test_core03_eat():
    L = lvl([(1, 2), (1, 3)], apples=[(2, 2), (4, 4)])
    r = step(L, initial_state(L), "RIGHT")
    assert r.event.ate_apple and r.state.head == (2, 2)
    assert r.state.segments == ((2, 2), (1, 2), (1, 3))
    assert r.state.apples == frozenset({(4, 4)})


# ---- T-CORE-04: exit before apples doesn't win; after -> WON
def test_core04_exit_early_not_won():
    L = lvl([(1, 2), (1, 3)], apples=[(4, 2)], exit=(2, 2))
    r = step(L, initial_state(L), "RIGHT")
    assert r.state.head == (2, 2) and r.state.status == Status.PLAYING and not r.event.won
    r = step(L, r.state, "RIGHT")
    r = step(L, r.state, "RIGHT")               # eat last apple at (4,2), adjacent to nothing yet
    assert r.event.ate_apple and r.state.status == Status.PLAYING
    st = replay(L, ["RIGHT", "RIGHT", "RIGHT", "UP", "LEFT", "LEFT", "DOWN"])
    assert st.status == Status.WON


# ---- T-CORE-05: WON beats TIMEOUT; invalid until max_steps -> truncated
def test_core05_win_on_last_step():
    L = demo_level()
    s = State(((3, 2), (2, 2), (1, 2), (1, 3)), frozenset(), L.max_steps - 1)
    r = step(L, s, "RIGHT")
    assert r.state.status == Status.WON and r.terminated and not r.truncated


def test_core05_invalid_until_timeout():
    L = demo_level()
    s = initial_state(L)
    for _ in range(L.max_steps):
        r = step(L, s, "DOWN")
        s = r.state
    assert r.truncated and not r.terminated and s.status == Status.TIMEOUT
    assert s.step_count == L.max_steps


def test_core05_valid_move_timeout():
    L = demo_level()
    s = State(L.segments, L.apples, L.max_steps - 1)
    r = step(L, s, "UP")
    assert r.truncated and r.event.timeout and r.state.status == Status.TIMEOUT


# ---- T-CORE-09: step after terminal
def test_core09_episode_finished():
    L = demo_level()
    st = replay(L, ["RIGHT"] * 3)
    with pytest.raises(GameError) as e:
        step(L, st, "UP")
    assert e.value.code == "EPISODE_FINISHED"


def test_invalid_action_enum():
    L = demo_level()
    for bad in ("NORTH", 4, -1, True, None):
        with pytest.raises(GameError) as e:
            step(L, initial_state(L), bad)
        assert e.value.code == "INVALID_ACTION"


# ---- T-CORE-10: invalid levels rejected
@pytest.mark.parametrize("kwargs", [
    dict(width=5, height=6),
    dict(width=4, height=4),
    dict(width=9, height=9),
    dict(segments=[(1, 1)]),
    dict(segments=[(1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]),
    dict(segments=[(1, 1), (2, 2)]),
    dict(segments=[(1, 1), (1, 2), (1, 1)]),
    dict(apples=[]),
    dict(apples=[(0, 0)] * 2),
    dict(apples=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]),
    dict(apples=[(1, 1)]),
    dict(walls=[(4, 4)]),
    dict(exit=(1, 2)),
    dict(exit=(9, 9)),
    dict(walls=[(1, 0), (0, 1), (2, 1)], segments=[(1, 1), (1, 2)]),  # boxed in
])
def test_core10_invalid_levels(kwargs):
    base = dict(width=5, height=5, walls=[], segments=[(1, 1), (1, 2)], apples=[(4, 4)], exit=(3, 3))
    base.update(kwargs)
    with pytest.raises(GameError) as e:
        make_level(**base)
    assert e.value.code == "INVALID_LEVEL"


def test_core10_unsupported_ruleset():
    with pytest.raises(GameError) as e:
        make_level(5, 5, [], [(1, 1), (1, 2)], [(4, 4)], (3, 3), ruleset_id="worm-grid-v2")
    assert e.value.code == "UNSUPPORTED_RULESET"


def test_core10_non_integer_coords():
    with pytest.raises(GameError):
        make_level(5, 5, [], [(1.0, 1), (1, 2)], [(4, 4)], (3, 3))


# ---- T-CORE-11: deadlock -> LOST
def test_core11_deadlock_lost():
    L = lvl([(1, 0), (2, 0)], walls=[(0, 1)])
    r = step(L, initial_state(L), "LEFT")
    assert r.state.status == Status.LOST and r.terminated and r.event.deadlock
    assert not any(valid_action_mask(L, r.state))


def test_reset_same_state():
    L = demo_level()
    assert initial_state(L) == initial_state(L)
    assert state_hash(initial_state(L)) == state_hash(initial_state(L))
