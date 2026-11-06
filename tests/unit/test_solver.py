"""T-SOLVER-01..04 (SPEC-06 §3)."""
import random

from tests.helpers import demo_level, random_level
from solver import EXHAUSTED, LIMIT, SOLVED, make_certificate, solve, verify_actions, verify_certificate
from worm_core import Status, initial_state, make_level, step


def reference_shortest(level, max_depth):
    """Independent iterative-deepening search over raw core.step (no shared successors/advance)."""
    frontier = {(initial_state(level).segments, initial_state(level).apples): initial_state(level)}
    seen = set(frontier)
    for depth in range(1, max_depth + 1):
        nxt = {}
        for st in frontier.values():
            for a in range(4):
                r = step(level, st, a)
                if not r.event.valid_action:
                    continue
                if r.state.status == Status.WON:
                    return depth
                k = (r.state.segments, r.state.apples)
                if r.state.status == Status.PLAYING and k not in seen:
                    seen.add(k)
                    nxt[k] = r.state
        if not nxt:
            return None
        frontier = nxt
    return None


def test_fixture_solution_and_count():
    res = solve(demo_level(), time_budget_ms=None)
    assert res.status == SOLVED and res.actions == ["RIGHT", "RIGHT", "RIGHT"] and res.optimal_steps == 3
    assert res.expanded_nodes == 24          # count convention: popped states incl. goal (SPEC-05 §2 note)


def test_solver01_matches_independent_reference():
    rng = random.Random(99)
    checked = 0
    while checked < 100:
        L = random_level(rng, size=5, max_apples=1)
        res = solve(L, node_budget=200_000, time_budget_ms=None)
        if res.status == LIMIT:
            continue
        ref = reference_shortest(L, 60)
        if res.status == SOLVED:
            assert ref == res.optimal_steps
        else:
            assert res.status == EXHAUSTED and ref is None
        checked += 1


def test_solver02_exhausted_only_when_queue_empty():
    # Worm boxed into a 2-cell corridor with the apple unreachable behind walls.
    L = make_level(5, 5, [(2, 0), (2, 1), (0, 2), (1, 2), (3, 3), (4, 3), (3, 4)], [(0, 0), (1, 0)], [(4, 4)], (0, 1))
    res = solve(L, time_budget_ms=None)
    assert res.status == EXHAUSTED and res.expanded_nodes < 150_000


def test_solver03_limit_is_not_a_verdict():
    rng = random.Random(5)
    for _ in range(50):
        L = random_level(rng, size=8)
        res = solve(L, node_budget=5, time_budget_ms=None)
        if res.status == LIMIT:
            assert res.actions is None and res.optimal_steps is None
            break
    else:
        raise AssertionError("expected at least one LIMIT")


def test_solver04_solved_paths_replay():
    rng = random.Random(3)
    n = 0
    for _ in range(200):
        L = random_level(rng)
        res = solve(L, node_budget=50_000, time_budget_ms=None)
        if res.status == SOLVED:
            assert verify_actions(L, res.actions) and len(res.actions) <= L.max_steps
            n += 1
    assert n > 20


def test_solver_deterministic():
    rng = random.Random(8)
    for _ in range(20):
        L = random_level(rng)
        a, b = solve(L, 50_000, None), solve(L, 50_000, None)
        assert (a.status, a.actions, a.expanded_nodes) == (b.status, b.actions, b.expanded_nodes)


def test_gen06_corrupted_certificate_rejected():
    L = demo_level()
    cert = make_certificate(L, solve(L, time_budget_ms=None))
    assert verify_certificate(L, cert)
    for mutate in (lambda c: c.update(actions=["RIGHT", "RIGHT", "UP"]),
                   lambda c: c.update(level_hash="0" * 64),
                   lambda c: c.update(optimal_steps=2),
                   lambda c: c.update(verified_by_replay=False),
                   lambda c: c.update(actions=["RIGHT"] * 4)):
        bad = dict(cert)
        mutate(bad)
        assert not verify_certificate(L, bad)

