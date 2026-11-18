"""Exact A* Puzzle Oracle, Counterfactual Mechanic Verifier, and Difficulty Analyzer."""

from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass
import heapq
import itertools

from worm_physics import DIRECTIONS, State, WormMap
from .physics import ACTIONS, BLOCKED_EVENTS, FATAL_EVENTS, LevelArena, validate_initial_level


@dataclass
class OracleResult:
    solvable: bool
    optimal_steps: int
    expanded_nodes: int
    exhausted_budget: bool
    min_portal_dist: int
    dead_end_ratio: float
    apples_total: int
    apples_eaten: int
    stones_total: int
    stones_moved: int
    mandatory_apples: int
    mandatory_stones: int
    unused_apples: int
    naive_greedy_wins: bool
    is_deceptive: bool
    direction_changes: int
    difficulty_score: float
    difficulty_tier: int
    tier_name: str
    actions: list[int]
    hints: list[dict[str, int]]

    def to_dict(self) -> dict:
        return asdict(self)


TIER_NAMES = {
    1: "Easy",
    2: "Medium",
    3: "Hard",
    4: "Extreme",
    5: "Grandmaster",
}


def solve_exact(
    board: WormMap,
    budget: int = 25000,
    max_depth: int = 250,
    root: State | None = None,
    avoid: set[State] | frozenset[State] = frozenset(),
) -> tuple[list[tuple[State, int, State, str]], int, bool, int, float]:
    """
    Admissible A* solver over canonical WormMap states.
    Returns (trace, expanded_nodes, exhausted_budget, min_portal_dist, dead_end_ratio).
    """
    start = root if root is not None else board.initial
    settled_start, init_ev = board.settle(start)
    if init_ev == "win":
        return [], 0, False, 0, 0.0
    if init_ev != "ok":
        return [], 0, False, 99, 1.0

    portals = tuple(board.portals)
    if not portals:
        return [], 0, False, 99, 1.0

    def h_adm(head: tuple[int, int]) -> int:
        hx, hy = head
        return min(abs(hx - gx) + (gy - hy if gy > hy else 0) for gx, gy in portals)

    def manhattan_portal(head: tuple[int, int]) -> int:
        hx, hy = head
        return min(abs(hx - gx) + abs(hy - gy) for gx, gy in portals)

    serial = itertools.count()
    h0 = h_adm(settled_start.body[0])
    min_dist = manhattan_portal(settled_start.body[0])
    # Heap item: (f_cost, tie_dist, g_cost, serial_id, state)
    heap: list[tuple[int, int, int, int, State]] = [
        (h0, min_dist, 0, next(serial), settled_start)
    ]
    g_best: dict[State, int] = {settled_start: 0}
    parent: dict[State, tuple[State, int, str] | None] = {settled_start: None}

    expanded = 0
    dead_branches = 0
    valid_branches = 0

    while heap and expanded < budget:
        f_val, _, depth, _, state = heapq.heappop(heap)
        if depth > g_best.get(state, max_depth + 1):
            continue
        if depth >= max_depth:
            continue

        expanded += 1
        legal_surviving = 0
        for action in ACTIONS:
            after, event = board.step(state, action)
            if event in BLOCKED_EVENTS:
                continue
            valid_branches += 1
            if event in FATAL_EVENTS:
                dead_branches += 1
                continue
            legal_surviving += 1
            d_port = manhattan_portal(after.body[0])
            if d_port < min_dist:
                min_dist = d_port

            if event == "win":
                # Reconstruct exact trajectory
                rev_trace: list[tuple[State, int, State, str]] = [(state, action, after, "win")]
                cur = state
                while parent[cur] is not None:
                    prev_state, prev_act, prev_ev = parent[cur]  # type: ignore[misc]
                    rev_trace.append((prev_state, prev_act, cur, prev_ev))
                    cur = prev_state
                rev_trace.reverse()
                ratio = dead_branches / max(1, valid_branches)
                return rev_trace, expanded, False, 0, ratio

            if after in avoid:
                continue
            nd = depth + 1
            if nd < g_best.get(after, max_depth + 1):
                g_best[after] = nd
                parent[after] = (state, action, "ok")
                nf = nd + h_adm(after.body[0])
                heapq.heappush(heap, (nf, d_port, nd, next(serial), after))

        if legal_surviving == 0:
            dead_branches += 1

    exhausted = bool(heap) and expanded >= budget
    ratio = dead_branches / max(1, valid_branches)
    return [], expanded, exhausted, min_dist, ratio


def check_naive_greedy_wins(arena: LevelArena, max_steps: int = 150) -> bool:
    """Test whether a myopic greedy-to-portal walk solves the map without real puzzle planning."""
    state = arena.board.initial
    seen = {state}
    for _ in range(max_steps):
        best_choice = None
        best_score = 1e9
        for action in ACTIONS:
            after, event = arena.board.step(state, action)
            if event == "win":
                return True
            if event != "ok" or after in seen:
                continue
            # Naive greedy score: minimize static distance to portal, slight preference for eating apples
            ate = 1 if len(after.apples) < len(state.apples) else 0
            score = arena.dist(after.body[0]) - 0.35 * ate
            if score < best_score:
                best_score = score
                best_choice = after
        if best_choice is None:
            return False
        state = best_choice
        seen.add(state)
    return False


def _count_turns(actions: list[int]) -> int:
    if len(actions) <= 1:
        return 0
    return sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])


def compute_difficulty(
    optimal_steps: int,
    mandatory_apples: int,
    mandatory_stones: int,
    stones_moved: int,
    hazards_count: int,
    dead_end_ratio: float,
    is_deceptive: bool,
    turns: int,
    expanded_nodes: int,
) -> tuple[float, int, str]:
    """Compute calibrated 0..100 difficulty score and 1..5 difficulty tier."""
    step_pts = min(38.0, optimal_steps * 0.68)
    mech_pts = min(24.0, mandatory_apples * 7.5 + mandatory_stones * 10.0 + stones_moved * 3.0)
    decep_pts = (11.0 if is_deceptive else 0.0) + min(10.0, dead_end_ratio * 24.0)
    hazard_pts = min(9.0, hazards_count * 1.1)
    search_pts = min(8.0, (expanded_nodes / 1500.0) * 4.0 + turns * 0.18)

    score = round(min(100.0, step_pts + mech_pts + decep_pts + hazard_pts + search_pts), 1)
    if score < 26.0 or optimal_steps < 13:
        tier = 1
    elif score < 44.0 or optimal_steps < 21:
        tier = 2
    elif score < 63.0 or optimal_steps < 31:
        tier = 3
    elif score < 80.0 or optimal_steps < 44:
        tier = 4
    else:
        tier = 5
    return score, tier, TIER_NAMES[tier]


def analyze_level(
    level: dict,
    budget: int = 25000,
    counterfactual_budget: int = 8000,
    verify_mandatory: bool = True,
) -> OracleResult:
    """Full exact verification, counterfactual mechanic audit, and difficulty rating for a map."""
    valid, _reason, board = validate_initial_level(level)
    if not valid or board is None:
        return OracleResult(
            solvable=False,
            optimal_steps=0,
            expanded_nodes=0,
            exhausted_budget=False,
            min_portal_dist=99,
            dead_end_ratio=1.0,
            apples_total=0,
            apples_eaten=0,
            stones_total=0,
            stones_moved=0,
            mandatory_apples=0,
            mandatory_stones=0,
            unused_apples=0,
            naive_greedy_wins=False,
            is_deceptive=False,
            direction_changes=0,
            difficulty_score=0.0,
            difficulty_tier=0,
            tier_name="Invalid",
            actions=[],
            hints=[],
        )

    trace, expanded, exhausted, min_dist, dead_ratio = solve_exact(board, budget=budget)
    apples_total = len(board.initial.apples)
    stones_total = len(board.initial.stones)

    if not trace:
        return OracleResult(
            solvable=False,
            optimal_steps=0,
            expanded_nodes=expanded,
            exhausted_budget=exhausted,
            min_portal_dist=min_dist,
            dead_end_ratio=round(dead_ratio, 4),
            apples_total=apples_total,
            apples_eaten=0,
            stones_total=stones_total,
            stones_moved=0,
            mandatory_apples=0,
            mandatory_stones=0,
            unused_apples=apples_total,
            naive_greedy_wins=False,
            is_deceptive=False,
            direction_changes=0,
            difficulty_score=0.0,
            difficulty_tier=0,
            tier_name="Unsolvable",
            actions=[],
            hints=[],
        )

    final_state = trace[-1][2]
    eaten_set = board.initial.apples - final_state.apples
    apples_eaten = len(eaten_set)
    unused_apples = apples_total - apples_eaten

    # Count distinct stone push/drop interactions along optimal path
    stones_moved = sum(1 for s, _, ns, _ in trace if s.stones != ns.stones)

    mandatory_apples = 0
    mandatory_stones = 0
    if verify_mandatory:
        # Counterfactual removal check for each eaten apple
        for apple_pos in eaten_set:
            mod_board = copy(board)
            mod_board.initial = State(
                board.initial.body,
                board.initial.apples - {apple_pos},
                board.initial.stones,
            )
            settled, ev = mod_board.settle(mod_board.initial)
            if ev != "ok":
                mandatory_apples += 1
                continue
            mod_board.initial = settled
            cf_trace, _, cf_exh, _, _ = solve_exact(mod_board, budget=counterfactual_budget)
            if (not cf_trace and not cf_exh) or (cf_trace and len(cf_trace) >= len(trace) + 4):
                mandatory_apples += 1
            elif not cf_trace and cf_exh:
                mandatory_apples += 1

        # Counterfactual check for stones: either needed as support/bridge OR blocks path until pushed
        for stone_pos in board.initial.stones:
            # Test 1: remove stone (checks if stone was required as bridge/support)
            rem_board = copy(board)
            rem_board.initial = State(
                board.initial.body,
                board.initial.apples,
                board.initial.stones - {stone_pos},
            )
            settled_r, ev_r = rem_board.settle(rem_board.initial)
            needed_as_bridge = False
            if ev_r != "ok":
                needed_as_bridge = True
            else:
                rem_board.initial = settled_r
                cf_rem, _, _, _, _ = solve_exact(rem_board, budget=counterfactual_budget)
                if not cf_rem or len(cf_rem) >= len(trace) + 4:
                    needed_as_bridge = True

            if needed_as_bridge:
                mandatory_stones += 1
                continue

            # Test 2: freeze stone into static wall (checks if pushing the stone was mandatory!)
            wall_board = copy(board)
            wall_board.walls = board.walls | {stone_pos}
            wall_board.fixed = board.fixed | {stone_pos}
            wall_board.support = board.support | {stone_pos}
            wall_board.initial = State(
                board.initial.body,
                board.initial.apples,
                board.initial.stones - {stone_pos},
            )
            cf_wall, _, _, _, _ = solve_exact(wall_board, budget=counterfactual_budget)
            if not cf_wall or len(cf_wall) >= len(trace) + 3:
                mandatory_stones += 1

    arena = LevelArena(board, level_dict=level)
    naive_wins = check_naive_greedy_wins(arena)
    is_deceptive = not naive_wins

    actions = [act for _, act, _, _ in trace]
    hints = [{"x": int(ns.body[0][0]), "y": int(ns.body[0][1]), "d": int(act)} for _, act, ns, _ in trace]
    turns = _count_turns(actions)

    diff_score, tier, tier_name = compute_difficulty(
        optimal_steps=len(trace),
        mandatory_apples=mandatory_apples,
        mandatory_stones=mandatory_stones,
        stones_moved=stones_moved,
        hazards_count=len(board.hazards),
        dead_end_ratio=dead_ratio,
        is_deceptive=is_deceptive,
        turns=turns,
        expanded_nodes=expanded,
    )

    return OracleResult(
        solvable=True,
        optimal_steps=len(trace),
        expanded_nodes=expanded,
        exhausted_budget=False,
        min_portal_dist=0,
        dead_end_ratio=round(dead_ratio, 4),
        apples_total=apples_total,
        apples_eaten=apples_eaten,
        stones_total=stones_total,
        stones_moved=stones_moved,
        mandatory_apples=mandatory_apples,
        mandatory_stones=mandatory_stones,
        unused_apples=unused_apples,
        naive_greedy_wins=naive_wins,
        is_deceptive=is_deceptive,
        direction_changes=turns,
        difficulty_score=diff_score,
        difficulty_tier=tier,
        tier_name=tier_name,
        actions=actions,
        hints=hints,
    )
