"""Automated verification tests for WormLabRL Physics, Exact Oracle, Continual RL Solver, and PCGRL."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worm_physics import State, WormMap
from worm_ai.oracle import analyze_level
from worm_ai.pcgrl import ProgressivePCGRLGenerator
from worm_ai.physics import (
    STRICT_RULES,
    LevelArena,
    load_dataset,
    load_levels_json,
    step_strict,
    step_strict_trace,
    validate_initial_level,
)
from worm_ai.trainer import ContinualWormTrainer


def test_strict_gravity_bomb_and_spike_lethal():
    """Regression test for Screenshot 1 (A107 bomb) and Screenshot 2 (G1003 spike + deadlock)."""
    lvs = {f"{p}{lv['id']}": lv for p, lv in load_levels_json()}
    b107 = WormMap(lvs["A107"], rules=STRICT_RULES)

    # Screenshot 1: worm at ((4,1),(3,1),(3,2),(3,3)) above bomb (3,0) MUST NOT float — must fall & die ('trap')!
    hovering_above_bomb = State(
        body=((4, 1), (3, 1), (3, 2), (3, 3)),
        apples=b107.initial.apples,
        stones=b107.initial.stones,
    )
    frames_107, ev_107 = b107.settle_trace(hovering_above_bomb)
    assert ev_107 == "trap", f"Expected instant 'trap' on bomb (3,0), got {ev_107}"
    assert (3, 0) in frames_107[-1].body, "Worm must fall onto bomb (3,0)"

    # Screenshot 2: vertical worm ((7,5),(7,4),(7,3)) above spike (7,2) MUST fall onto spike & die ('trap')!
    spike_lv = {
        "width": 22,
        "height": 12,
        "bricks": [
            {"x": 6, "y": 2, "id": 1}, {"x": 6, "y": 3, "id": 1},
            {"x": 6, "y": 4, "id": 1}, {"x": 6, "y": 5, "id": 1},
            {"x": 7, "y": 1, "id": 1}, {"x": 7, "y": 2, "id": 3},
            {"x": 7, "y": 5, "id": 5}, {"x": 7, "y": 4, "id": 6}, {"x": 7, "y": 3, "id": 6},
            {"x": 5, "y": 2, "id": 8},
        ],
    }
    valid_s2, reason_s2, _ = validate_initial_level(spike_lv)
    assert valid_s2 is False and "trap" in reason_s2, f"Map starting above spike must fail frame-0 check: {reason_s2}"

    # Deadlock check: moving into a 1x1 pocket surrounded by walls/spikes with 0 surviving moves -> 'deadlock'!
    pocket_lv = {
        "width": 22,
        "height": 12,
        "bricks": [
            {"x": 4, "y": 3, "id": 1}, {"x": 5, "y": 3, "id": 1}, {"x": 6, "y": 3, "id": 1}, {"x": 7, "y": 3, "id": 1},
            {"x": 7, "y": 4, "id": 3},  # spike on right
            {"x": 5, "y": 5, "id": 1}, {"x": 6, "y": 5, "id": 1},  # ceiling blocking up
            {"x": 4, "y": 4, "id": 1},  # left wall
            {"x": 5, "y": 4, "id": 5}, {"x": 5, "y": 6, "id": 6},  # wait: valid connected worm below
            {"x": 15, "y": 5, "id": 8},
        ],
    }
    pocket_lv["bricks"] = [
        {"x": 4, "y": 3, "id": 1}, {"x": 5, "y": 3, "id": 1}, {"x": 6, "y": 3, "id": 1}, {"x": 7, "y": 3, "id": 1},
        {"x": 8, "y": 4, "id": 1},  # right wall
        {"x": 7, "y": 5, "id": 1},  # ceiling over (7,4)
        {"x": 6, "y": 4, "id": 5}, {"x": 5, "y": 4, "id": 6}, {"x": 4, "y": 4, "id": 6},
        {"x": 15, "y": 5, "id": 8},
    ]
    b_pock = WormMap(pocket_lv, rules=STRICT_RULES)
    # Moving right (action=2) puts head at (7,4), body at ((7,4),(6,4),(5,4)):
    # From (7,4): Right=(8,4) wall, Up=(7,5) wall, Down=(7,3) wall, Left=(6,4) neck -> 0 surviving moves!
    _, ev_pock = step_strict(b_pock, b_pock.initial, 2)
    assert ev_pock == "deadlock", f"Expected immediate 'deadlock', got {ev_pock}"


def test_oracle_exactness_and_counterfactuals():
    splits = load_dataset()
    # Pick A15 (2 mandatory apples + 2 mandatory stones)
    row_a15 = next(r for s in ("train", "val", "test") for r in splits[s] if r[0]["name"] == "A15")
    res = analyze_level(row_a15[0]["level"], budget=30000)
    assert res.solvable is True
    assert res.optimal_steps <= row_a15[0]["hint_steps"]
    assert res.mandatory_apples >= 1
    assert res.mandatory_stones >= 1
    assert res.unused_apples == 0


def test_pcgrl_escalating_solvability():
    gen = ProgressivePCGRLGenerator(seed=123)
    scores = []
    floor = 12.0
    for tier in (1, 2, 3, 4, 5):
        lv, ores = gen.generate_verified_map(target_tier=tier, min_difficulty_floor=floor, max_attempts=40)
        valid, reason, board = validate_initial_level(lv)
        assert valid and board is not None, f"Invalid generated map: {reason}"
        assert len(board.bodies) == 1, "Ambiguous worm body reconstruction"
        assert ores.solvable is True, "Generated map must be 100% solvable"
        assert ores.unused_apples == 0, "Generated map must have zero fake/unused apples"
        scores.append(ores.difficulty_score)
        floor = ores.difficulty_score + 1.5

    assert scores[-1] > scores[0] + 20.0, f"Expected significant difficulty increase across tiers, got {scores}"


def test_continual_stage_mastery_and_retention():
    splits = load_dataset()
    trainer = ContinualWormTrainer(seed=73)
    r1 = next(r for r in splits["train"] if r[0]["name"] == "A2")
    r2 = next(r for r in splits["train"] if r[0]["name"] == "A6")

    s1 = trainer.play_and_self_train_stage(r1[1], ppo_rounds=2, envs=6, horizon=24, rescue_budget=1500)
    assert s1["after_train_won"] is True

    s2 = trainer.play_and_self_train_stage(r2[1], ppo_rounds=2, envs=6, horizon=24, rescue_budget=1500)
    assert s2["after_train_won"] is True

    # Ensure r1 is still solved zero-shot after learning r2!
    check_r1 = trainer.play_episode(r1[1], greedy=True)
    assert check_r1.won is True, "Catastrophic forgetting occurred on previously mastered stage A2!"


if __name__ == "__main__":
    test_strict_gravity_bomb_and_spike_lethal()
    print("[PASS] test_strict_gravity_bomb_and_spike_lethal")
    test_oracle_exactness_and_counterfactuals()
    print("[PASS] test_oracle_exactness_and_counterfactuals")
    test_pcgrl_escalating_solvability()
    print("[PASS] test_pcgrl_escalating_solvability")
    test_continual_stage_mastery_and_retention()
    print("[PASS] test_continual_stage_mastery_and_retention")
    print("ALL TESTS PASSED!")
