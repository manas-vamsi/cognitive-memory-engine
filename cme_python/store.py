"""Belief registry — persistent storage for beliefs.

SQLite via the stdlib. The full belief round-trips through Pydantic JSON in a
`data` column; the columns beside it exist only so we can filter and sort
without deserialising every row.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from cme_python.models import Belief

SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    id         TEXT PRIMARY KEY,
    statement  TEXT NOT NULL,
    confidence REAL NOT NULL,
    source     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    data       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence);
"""


class BeliefStore:
    """Persistent registry of beliefs. Use as a context manager or call close()."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI runs sync endpoints on a threadpool, so the connection is
        # reached from whichever worker thread serves the request. SQLite
        # refuses that by default; one connection plus one lock is the smallest
        # correct answer.
        #
        # ponytail: a single global lock serialises every query. Right while
        # requests are cheap; move to a per-thread connection or a pool if
        # concurrent reads start queueing.
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def __enter__(self) -> BeliefStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._db:
            yield self._db

    def _read(self, sql: str, params: Sequence[object] = ()) -> list[sqlite3.Row]:
        """Rows are materialised inside the lock; a lazy cursor would escape it."""
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # --- registry operations ----------------------------------------------

    def save(self, belief: Belief) -> Belief:
        """Insert or update. Saving is idempotent on belief id."""
        with self._write() as db:
            db.execute(
                """INSERT INTO beliefs (id, statement, confidence, source, created_at, updated_at, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       statement=excluded.statement, confidence=excluded.confidence,
                       source=excluded.source, updated_at=excluded.updated_at, data=excluded.data""",
                (
                    belief.id,
                    belief.statement,
                    belief.confidence,
                    str(belief.source),
                    belief.created_at.isoformat(),
                    belief.updated_at.isoformat(),
                    belief.model_dump_json(),
                ),
            )
        return belief

    def save_all(self, beliefs: list[Belief]) -> int:
        for b in beliefs:
            self.save(b)
        return len(beliefs)

    def get(self, belief_id: str) -> Belief | None:
        rows = self._read("SELECT data FROM beliefs WHERE id = ?", (belief_id,))
        return Belief.model_validate_json(rows[0]["data"]) if rows else None

    def delete(self, belief_id: str) -> bool:
        with self._write() as db:
            return db.execute("DELETE FROM beliefs WHERE id = ?", (belief_id,)).rowcount > 0

    def all(self, *, min_confidence: float = 0.0, limit: int | None = None) -> list[Belief]:
        """Beliefs above a confidence floor, strongest first."""
        sql = "SELECT data FROM beliefs WHERE confidence >= ? ORDER BY confidence DESC"
        params: list[object] = [min_confidence]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [Belief.model_validate_json(r["data"]) for r in self._read(sql, params)]

    def search(self, text: str, *, limit: int = 20) -> list[Belief]:
        """Substring match on the statement.

        ponytail: LIKE scan, not semantic search — the Evidence Engine's vector
        store is where real retrieval lands. Swap in FTS5 if this gets slow.
        """
        rows = self._read(
            "SELECT data FROM beliefs WHERE statement LIKE ? ORDER BY confidence DESC LIMIT ?",
            (f"%{text}%", limit),
        )
        return [Belief.model_validate_json(r["data"]) for r in rows]

    def prune_dead(self, threshold: float) -> int:
        """Drop disproven beliefs out of the registry. Returns rows removed."""
        with self._write() as db:
            return db.execute("DELETE FROM beliefs WHERE confidence <= ?", (threshold,)).rowcount

    def __len__(self) -> int:
        return self._read("SELECT COUNT(*) AS n FROM beliefs")[0]["n"]
