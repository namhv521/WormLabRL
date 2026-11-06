"""SQLite persistence for published levels + certificates (SPEC-04 §5)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional

from worm_core import level_from_dict

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "wormlab.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS levels (
  level_id TEXT PRIMARY KEY,
  ruleset_id TEXT NOT NULL,
  canonical_hash TEXT NOT NULL,
  level_hash TEXT NOT NULL,
  tier TEXT NOT NULL,
  difficulty_score REAL NOT NULL,
  split TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('CANDIDATE','PUBLISHED')),
  definition TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  UNIQUE (ruleset_id, canonical_hash)
);
CREATE TABLE IF NOT EXISTS certificates (
  level_id TEXT PRIMARY KEY REFERENCES levels(level_id),
  certificate TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_levels_pool ON levels(status, split, tier, difficulty_score);
"""


class LevelStore:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def publish(self, record: dict, certificate: dict) -> bool:
        """Insert as CANDIDATE, attach certificate, then flip to PUBLISHED in one transaction.

        Returns False when the canonical hash already exists (dedup).
        """
        from solver import verify_certificate
        if not verify_certificate(level_from_dict(record), certificate):
            from worm_core import GameError
            raise GameError("INTERNAL_VERIFICATION_FAILED", f"certificate rejected for {record['level_id']}")
        try:
            with self.tx() as c:
                c.execute("INSERT INTO levels(level_id, ruleset_id, canonical_hash, level_hash, tier,"
                          " difficulty_score, split, status, definition) VALUES (?,?,?,?,?,?,?,'CANDIDATE',?)",
                          (record["level_id"], record["ruleset_id"], record["canonical_hash"],
                           record["level_hash"], record["difficulty_label"], record["difficulty_score"],
                           record["split"], json.dumps(record, sort_keys=True)))
                c.execute("INSERT INTO certificates(level_id, certificate) VALUES (?,?)",
                          (record["level_id"], json.dumps(certificate, sort_keys=True)))
                c.execute("UPDATE levels SET status='PUBLISHED' WHERE level_id=?", (record["level_id"],))
            return True
        except sqlite3.IntegrityError:
            return False

    def get(self, level_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT definition FROM levels WHERE level_id=? AND status='PUBLISHED'",
                                (level_id,)).fetchone()
        return json.loads(row["definition"]) if row else None

    def certificate(self, level_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT c.certificate FROM certificates c JOIN levels l USING(level_id)"
                                " WHERE level_id=? AND l.status='PUBLISHED'", (level_id,)).fetchone()
        return json.loads(row["certificate"]) if row else None

    def pool(self, tier: str, split: str = "pool", exclude: Iterable[str] = ()) -> List[dict]:
        ex = list(exclude)
        q = ("SELECT definition FROM levels WHERE status='PUBLISHED' AND split=? AND tier=?"
             + (f" AND level_id NOT IN ({','.join('?' * len(ex))})" if ex else "")
             + " ORDER BY difficulty_score, level_id")
        return [json.loads(r["definition"]) for r in self.conn.execute(q, (split, tier, *ex))]

    def counts(self, split: str = "pool") -> dict:
        rows = self.conn.execute("SELECT tier, COUNT(*) n FROM levels WHERE status='PUBLISHED' AND split=?"
                                 " GROUP BY tier", (split,)).fetchall()
        return {r["tier"]: r["n"] for r in rows}

    def canonical_hashes(self) -> set:
        return {r[0] for r in self.conn.execute("SELECT canonical_hash FROM levels")}

    def close(self):
        self.conn.close()
