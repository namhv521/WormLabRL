"""T-RL-03 / T-GEN-01/02 on the committed manifests (skipped if not built)."""
from pathlib import Path

import pytest

from generator.build import load_manifest
from solver import verify_actions, solve
from worm_core import level_from_dict

M = Path(__file__).resolve().parents[2] / "data" / "manifests"
SPLITS = ("train", "validation", "test")
pytestmark = pytest.mark.skipif(not (M / "test-v1.json").exists(), reason="manifests not built")


@pytest.fixture(scope="module")
def manifests():
    return {s: load_manifest(M / f"{s}-v1.json") for s in SPLITS}


def test_rl03_splits_disjoint(manifests):
    hs = {s: {r["canonical_hash"] for r in m["levels"]} for s, m in manifests.items()}
    for s in SPLITS:
        assert len(hs[s]) == len(manifests[s]["levels"])
    assert not hs["train"] & hs["validation"]
    assert not hs["train"] & hs["test"]
    assert not hs["validation"] & hs["test"]


def test_test_manifest_size(manifests):
    assert manifests["test"]["counts"] == {"easy": 100, "medium": 100, "hard": 100}


@pytest.mark.slow
def test_gen01_every_manifest_level_resolves(manifests):
    # Re-derive certificates independently from the serialized definition (AC-GEN-02).
    n = 0
    for m in manifests.values():
        for r in m["levels"]:
            L = level_from_dict(r)
            res = solve(L, time_budget_ms=None)
            assert res.optimal_steps == r["optimal_steps"] and verify_actions(L, res.actions)
            n += 1
    assert n >= 1000
