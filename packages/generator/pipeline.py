"""Candidate -> BFS -> replay -> grade -> dedup pipeline (SPEC-03 §3)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

from solver import DEFAULT_NODE_BUDGET, SOLVED, make_certificate, solve
from worm_core import Level, canonical_hash, level_hash

from .candidate import candidate_seed, generate_candidate
from .difficulty import DIFFICULTY_VERSION, difficulty_score, tier_of


@dataclass
class CertifiedLevel:
    level: Level
    tier: str
    score: float
    level_hash: str
    canonical_hash: str
    certificate: dict

    @property
    def level_id(self) -> str:
        return f"{self.tier}-{self.canonical_hash[:12]}"


@dataclass
class BatchReport:
    tier: str
    requested: int
    candidates: int = 0
    accepted: int = 0
    rejections: Counter = field(default_factory=Counter)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.candidates if self.candidates else 0.0

    def to_dict(self) -> dict:
        return {"tier": self.tier, "requested": self.requested, "candidates": self.candidates,
                "accepted": self.accepted, "acceptance_rate": round(self.acceptance_rate, 4),
                "rejections": dict(sorted(self.rejections.items()))}


def certify(level: Level, node_budget: int = DEFAULT_NODE_BUDGET,
            time_budget_ms: Optional[int] = None):
    """Return ``(CertifiedLevel, None)`` or ``(None, reject_reason)``.

    The default disables the wall-clock budget so certification and difficulty
    are reproducible across machines; the node budget still bounds work.
    """
    res = solve(level, node_budget=node_budget, time_budget_ms=time_budget_ms)
    if res.status != SOLVED:
        return None, res.status
    cert = make_certificate(level, res)
    cert["difficulty_version"] = DIFFICULTY_VERSION
    score = difficulty_score(res.optimal_steps, res.expanded_nodes)
    return CertifiedLevel(level, tier_of(score), score, level_hash(level), canonical_hash(level), cert), None


def generate_batch(tier: str, count: int, *, base_seed: int = 0, namespace: str = "pool",
                   max_candidates: int = 5000, seen: Optional[Set[str]] = None,
                   node_budget: int = DEFAULT_NODE_BUDGET) -> tuple[List[CertifiedLevel], BatchReport]:
    """Generate up to ``count`` certified, deduplicated levels of exactly ``tier``.

    ``seen`` holds canonical hashes already used anywhere (other splits, DB);
    it is updated in place. Candidate order is a pure function of the arguments.
    """
    seen = set() if seen is None else seen
    out: List[CertifiedLevel] = []
    rep = BatchReport(tier, count)
    for i in range(max_candidates):
        if len(out) >= count:
            break
        rep.candidates += 1
        level = generate_candidate(tier, candidate_seed(namespace, tier, base_seed, i))
        if level is None:
            rep.rejections["CONSTRUCTION_FAILED"] += 1
            continue
        ch = canonical_hash(level)
        if ch in seen:
            rep.rejections["DUPLICATE"] += 1
            continue
        cl, reason = certify(level, node_budget=node_budget)
        if cl is None:
            rep.rejections[reason] += 1
            continue
        if cl.tier != tier:
            rep.rejections[f"WRONG_TIER_{cl.tier.upper()}"] += 1
            continue
        seen.add(ch)
        out.append(cl)
        rep.accepted += 1
    return out, rep


def certified_record(cl: CertifiedLevel, split: str) -> dict:
    """Self-contained manifest/DB record for a certified level."""
    from dataclasses import replace

    from worm_core import level_to_dict
    d = level_to_dict(replace(cl.level, level_id=cl.level_id))
    d.update({"verified": True, "difficulty_label": cl.tier, "difficulty_score": cl.score,
              "difficulty_version": DIFFICULTY_VERSION, "level_hash": cl.level_hash,
              "canonical_hash": cl.canonical_hash, "split": split,
              "optimal_steps": cl.certificate["optimal_steps"]})
    return d


def iter_hashes(records: Iterable[dict]) -> Set[str]:
    return {r["canonical_hash"] for r in records}
