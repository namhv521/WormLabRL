"""Build locked train/validation/test manifests and the demo pool.

    python -m generator.build --out data/manifests --db data/wormlab.db

Splits are generated in a fixed order (test -> validation -> train -> pool) and
share one ``seen`` canonical-hash set, so no puzzle (up to rotation/reflection)
can appear in two splits (SPEC-02 §5.1). Existing manifests are never
overwritten unless ``--force`` is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from worm_core import RULESET_ID, canonical_json, level_from_dict

from .candidate import GENERATOR_VERSION
from .difficulty import DIFFICULTY_VERSION, TIERS
from .pipeline import certified_record, certify, generate_batch

DATASET_VERSION = "v1"
DEFAULT_SIZES = {
    "test": {"easy": 100, "medium": 100, "hard": 100},
    "validation": {"easy": 50, "medium": 50, "hard": 50},
    "train": {"easy": 600, "medium": 600, "hard": 300},
    "pool": {"easy": 30, "medium": 30, "hard": 30},
}
SPLIT_SEEDS = {"test": 3_000, "validation": 2_000, "train": 1_000, "pool": 4_000}
MAX_CANDIDATES = {"easy": 5_000, "medium": 5_000, "hard": 20_000}


def dataset_hash(records) -> str:
    return hashlib.sha256(canonical_json(sorted(r["canonical_hash"] for r in records)).encode()).hexdigest()


def manifest_path(out: Path, split: str) -> Path:
    return out / f"{split}-{DATASET_VERSION}.json"


def load_manifest(path: Path | str) -> dict:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if dataset_hash(m["levels"]) != m["dataset_hash"]:
        raise ValueError(f"dataset_hash mismatch in {path}")
    return m


def build_split(split: str, sizes: dict, seen: set, log=print):
    records, certs, reports = [], {}, {}
    for tier in TIERS:
        if not sizes.get(tier):
            continue
        t0 = time.time()
        levels, rep = generate_batch(tier, sizes[tier], base_seed=SPLIT_SEEDS[split], namespace=split,
                                     max_candidates=MAX_CANDIDATES[tier], seen=seen)
        for cl in levels:
            rec = certified_record(cl, split)
            records.append(rec)
            certs[rec["level_id"]] = cl.certificate
        reports[tier] = rep.to_dict()
        short = "" if rep.accepted >= sizes[tier] else f"  ** SHORT by {sizes[tier] - rep.accepted} **"
        log(f"  {split:<10} {tier:<6} {rep.accepted:>4}/{sizes[tier]:<4} from {rep.candidates:>5} candidates"
            f" ({time.time() - t0:.1f}s){short}")
    return records, certs, reports


def write_manifest(path: Path, split: str, records, reports):
    path.write_text(json.dumps({
        "dataset_version": DATASET_VERSION, "split": split, "ruleset_id": RULESET_ID,
        "generator_version": GENERATOR_VERSION, "difficulty_version": DIFFICULTY_VERSION,
        "dataset_hash": dataset_hash(records),
        "counts": {t: sum(r["difficulty_label"] == t for r in records) for t in TIERS},
        "generation_report": reports, "levels": records,
    }, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def publish_demo_fixture(store, root: Path):
    rec = json.loads((root / "data" / "levels" / "demo-5x5-001.json").read_text(encoding="utf-8"))
    cl, reason = certify(level_from_dict(rec))
    if cl is None:
        raise RuntimeError(f"demo fixture not certifiable: {reason}")
    out = certified_record(cl, "pool")
    out["level_id"] = rec["level_id"]
    out["seed"] = rec["seed"]
    store.publish(out, cl.certificate)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/manifests")
    p.add_argument("--db", default="data/wormlab.db")
    p.add_argument("--scale", type=float, default=1.0, help="multiply split sizes (smoke runs)")
    p.add_argument("--force", action="store_true", help="overwrite existing manifests")
    a = p.parse_args(argv)

    from storage import LevelStore
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    store = LevelStore(a.db)
    # Demo pool lives in the DB only; publish the fixture first so no split can contain it.
    if store.get("demo-5x5-001") is None:
        publish_demo_fixture(store, root)
    seen = set(store.canonical_hashes())

    for split in ("test", "validation", "train"):
        path = manifest_path(out, split)
        if path.exists() and not a.force:
            m = load_manifest(path)
            seen |= {r["canonical_hash"] for r in m["levels"]}
            print(f"  {split:<10} locked manifest kept ({len(m['levels'])} levels, {m['dataset_hash'][:12]})")
            continue
        sizes = {t: max(1, round(n * a.scale)) for t, n in DEFAULT_SIZES[split].items()}
        records, _, reports = build_split(split, sizes, seen)
        write_manifest(path, split, records, reports)

    have = store.counts("pool")
    need = {t: max(0, DEFAULT_SIZES["pool"][t] - have.get(t, 0)) for t in TIERS}
    records, certs, _ = build_split("pool", need, seen)
    published = sum(store.publish(r, certs[r["level_id"]]) for r in records)
    print(f"  pool published +{published}; totals {store.counts('pool')}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
