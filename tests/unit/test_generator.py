"""T-GEN-01..06 (SPEC-06 §3) on small, fast batches; full-size checks run on manifests."""
import pytest

from generator import (BANDS, candidate_seed, certified_record, certify, difficulty_score, generate_batch,
                       generate_candidate, tier_of)
from solver import verify_certificate
from storage import LevelStore
from tests.helpers import demo_level
from worm_core import canonical_hash, level_from_dict, level_hash, transformed_levels


@pytest.fixture(scope="module")
def batches():
    seen = set()
    return {t: generate_batch(t, n, base_seed=1, namespace="t", seen=seen)
            for t, n in (("easy", 40), ("medium", 30), ("hard", 4))}


def test_gen01_all_published_certified(batches):
    for tier, (levels, rep) in batches.items():
        assert levels, f"no {tier} levels"
        for cl in levels:
            assert cl.certificate["verified_by_replay"] is True
            assert cl.certificate["optimal_steps"] <= cl.level.max_steps
            assert verify_certificate(level_from_dict(certified_record(cl, "x")), cl.certificate)


def test_gen02_no_canonical_duplicates(batches):
    hashes = [cl.canonical_hash for levels, _ in batches.values() for cl in levels]
    assert len(hashes) == len(set(hashes))


def test_gen03_label_matches_score(batches):
    for tier, (levels, _) in batches.items():
        for cl in levels:
            c = cl.certificate
            assert cl.score == difficulty_score(c["optimal_steps"], c["expanded_nodes"])
            lo, hi = BANDS[tier]
            assert lo <= cl.score < hi or (tier == "hard" and cl.score == 1.0)
            assert cl.tier == tier == tier_of(cl.score)


def test_gen04_seed_stream_deterministic():
    a = [generate_candidate("medium", candidate_seed("x", "medium", 7, i)) for i in range(30)]
    b = [generate_candidate("medium", candidate_seed("x", "medium", 7, i)) for i in range(30)]
    assert [L and level_hash(L) for L in a] == [L and level_hash(L) for L in b]
    assert candidate_seed("x", "easy", 0, 0) != candidate_seed("y", "easy", 0, 0)


def test_gen05_exhausted_budget_reports_short():
    levels, rep = generate_batch("hard", 50, base_seed=2, namespace="short", max_candidates=10)
    assert len(levels) < 50 and rep.candidates == 10
    assert sum(rep.rejections.values()) + rep.accepted == 10


def test_difficulty_formula_fixture():
    # SPEC-05 §2: fixture scores 0.1568 with 24 expansions and 3 optimal steps.
    cl, _ = certify(demo_level())
    assert cl.certificate["expanded_nodes"] == 24 and cl.score == 0.1568 and cl.tier == "easy"
    assert tier_of(0.3499) == "easy" and tier_of(0.35) == "medium" and tier_of(0.65) == "hard"


def test_canonical_hash_symmetry_invariant():
    L = demo_level()
    hs = {canonical_hash(v) for v in transformed_levels(L)}
    assert hs == {canonical_hash(L)}
    assert len({level_hash(v) for v in transformed_levels(L)}) > 1


def test_store_publish_dedup_and_reject(tmp_path):
    store = LevelStore(tmp_path / "t.db")
    cl, _ = certify(demo_level())
    rec = certified_record(cl, "pool")
    assert store.publish(rec, cl.certificate)
    assert store.get(rec["level_id"])["verified"] is True
    # Symmetric variant -> same canonical hash -> blocked by UNIQUE (AC-GEN-05).
    variant = next(v for v in transformed_levels(cl.level) if level_hash(v) != cl.level_hash)
    cl2, _ = certify(variant)
    rec2 = certified_record(cl2, "pool")
    rec2["level_id"] = "other-id"
    assert not store.publish(rec2, cl2.certificate)
    bad = dict(cl.certificate, actions=["UP"])
    rec3 = dict(rec, level_id="bad")
    with pytest.raises(Exception):
        store.publish(rec3, bad)
    assert store.get("bad") is None
