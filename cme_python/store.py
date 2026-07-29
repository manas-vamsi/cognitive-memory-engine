"""Belief registry — persistent storage for beliefs.

SQLite by default; PostgreSQL when the spec's deployment story calls for it
(see `postgres_store.py`). The full belief round-trips through Pydantic JSON in
a `data` column, and the columns beside it exist only so we can filter and sort
without deserialising every row.

The two dialects share every statement below. Where they genuinely differ —
parameter style, `LIKE` case-sensitivity, schema introspection — the difference
is a class attribute or a hook, not a forked implementation.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from cme_python.models import Belief, MemoryTier

SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    id         TEXT PRIMARY KEY,
    statement  TEXT NOT NULL,
    confidence REAL NOT NULL,
    source     TEXT NOT NULL,
    tier       TEXT NOT NULL DEFAULT 'general',
    scope      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    superseded_by TEXT,
    data       TEXT NOT NULL
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence)",
    "CREATE INDEX IF NOT EXISTS idx_beliefs_tier ON beliefs(tier, scope)",
]
"""Created after migration — indexing a column an older table lacks would fail."""

ADDED_COLUMNS = {
    "tier": "TEXT NOT NULL DEFAULT 'general'",
    "scope": "TEXT",
    "superseded_by": "TEXT",
}
"""Columns introduced after the first release, applied to older databases."""

UPSERT = """INSERT INTO beliefs
       (id, statement, confidence, source, tier, scope, created_at, updated_at,
        superseded_by, data)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
   ON CONFLICT(id) DO UPDATE SET
       statement=excluded.statement,
       confidence=excluded.confidence,
       source=excluded.source,
       tier=excluded.tier,
       scope=excluded.scope,
       updated_at=excluded.updated_at,
       superseded_by=excluded.superseded_by,
       data=excluded.data"""


class BeliefStore:
    """Persistent registry of beliefs. Use as a context manager or call close()."""

    placeholder = "?"
    like = "LIKE"
    """SQLite's LIKE ignores case; Postgres's does not, so it overrides this.

    Not cosmetic: `search()` backs duplicate detection in the Belief Engine, so
    a case-sensitive LIKE would quietly stop recognising "Rust is fast" and
    "rust is fast" as the same claim on one backend but not the other.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._db = self._connect()
        self._init_schema()

    # --- dialect hooks -----------------------------------------------------

    def _connect(self):
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI runs sync endpoints on a threadpool, so a connection is
        # reached from whichever worker thread serves the request, and SQLite
        # refuses that by default.
        #
        # ponytail: a single global lock serialises every query, and it stays.
        # Giving each thread its own connection was measured twice and made
        # concurrent throughput ~15% *worse* — the work is CPU-bound Python
        # under the GIL, so the lock was never what threads were waiting on.
        # Scale out with processes and a shared store, not with threads.
        db = sqlite3.connect(self.path, check_same_thread=False)
        db.row_factory = sqlite3.Row
        if self.path != ":memory:":
            # A rollback journal copies every page it is about to change into a
            # second file and waits for that to reach the disk, per write. WAL
            # appends and syncs once: ~4.5x faster on `save`, which is the whole
            # of ingest. Reads are unchanged — measured, not assumed.
            #
            # Reads being unaffected is also why this is the only performance
            # change here: the store lock does not slow retrieval down either.
            db.execute("PRAGMA journal_mode=WAL")
            # WAL lets readers run beside the writer, but two writers still
            # collide — several processes on one registry is exactly the case
            # WAL invites. Waiting briefly is what a caller wants there;
            # failing instantly is not.
            db.execute("PRAGMA busy_timeout=5000")
        return db

    def _existing_columns(self) -> set[str]:
        return {row["name"] for row in self._db.execute("PRAGMA table_info(beliefs)")}

    def _sql(self, statement: str) -> str:
        """Translate the shared SQL into this dialect."""
        if self.placeholder != "?":
            statement = statement.replace("?", self.placeholder)
        if self.like != "LIKE":
            statement = statement.replace(" LIKE ", f" {self.like} ")
        return statement

    # --- plumbing ----------------------------------------------------------

    def _init_schema(self) -> None:
        with self._write() as db:
            db.execute(SCHEMA)
        self._migrate()
        with self._write() as db:
            for index in INDEXES:
                db.execute(index)

    def _migrate(self) -> None:
        """Bring an older database up to the current schema.

        A registry written before tiers existed is still somebody's accumulated
        memory. Adding the columns and backfilling from the stored JSON keeps it
        readable instead of crashing on the first query.
        """
        missing = [name for name in ADDED_COLUMNS if name not in self._existing_columns()]
        if not missing:
            return
        with self._write() as db:
            for name in missing:
                db.execute(f"ALTER TABLE beliefs ADD COLUMN {name} {ADDED_COLUMNS[name]}")
        # The JSON blob is the source of truth; the columns only mirror it.
        for row in self._read("SELECT id, data FROM beliefs"):
            belief = Belief.model_validate_json(row["data"])
            self._exec(
                "UPDATE beliefs SET tier = ?, scope = ? WHERE id = ?",
                (str(belief.tier), belief.scope, row["id"]),
            )

    def __enter__(self) -> BeliefStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _write(self) -> Iterator:
        with self._lock, self._db:
            yield self._db.cursor() if self.placeholder != "?" else self._db

    def _exec(self, sql: str, params: Sequence[object] = ()) -> int:
        """Run a statement, returning rows affected."""
        with self._write() as db:
            return db.execute(self._sql(sql), params).rowcount

    def _read(self, sql: str, params: Sequence[object] = ()) -> list[Mapping]:
        """Rows are materialised inside the lock; a lazy cursor would escape it."""
        with self._lock:
            cursor = self._db.cursor()
            cursor.execute(self._sql(sql), params)
            return cursor.fetchall()

    # --- registry operations ----------------------------------------------

    def save(self, belief: Belief) -> Belief:
        """Insert or update. Saving is idempotent on belief id.

        The write stamps `updated_at` itself. The lifecycle methods maintain it
        too, but a caller assigning `belief.statement = ...` directly does not —
        and indexes now use that stamp to decide what changed, so a belief could
        be rewritten and retrieval would keep answering from the old text. Make
        the write authoritative and the invariant cannot be forgotten.
        """
        self._exec(UPSERT, self._row(belief))
        return belief

    def _row(self, belief: Belief) -> tuple:
        """The belief as a row, stamping `updated_at` on the way past."""
        belief.updated_at = datetime.now(UTC)
        return (
            belief.id,
            belief.statement,
            belief.confidence,
            str(belief.source),
            str(belief.tier),
            belief.scope,
            belief.created_at.isoformat(),
            belief.updated_at.isoformat(),
            belief.superseded_by,
            belief.model_dump_json(),
        )

    def save_all(self, beliefs: list[Belief]) -> int:
        """Save many beliefs in one transaction.

        Not a loop over `save`. A transaction ends in a disk sync, and a sync
        costs about the same whether it is flushing one row or a thousand — so
        saving a document's worth of beliefs one at a time paid that price per
        belief. Ingest is the caller that feels it: it produces beliefs in
        batches and used to commit each one separately.

        All or nothing, which is also the more correct behaviour: a document
        half-ingested because the process died between two of its claims leaves
        a registry nobody can reason about.
        """
        if not beliefs:
            return 0
        rows = [self._row(b) for b in beliefs]
        with self._write() as db:
            db.executemany(self._sql(UPSERT), rows)
        return len(beliefs)

    def get(self, belief_id: str) -> Belief | None:
        rows = self._read("SELECT data FROM beliefs WHERE id = ?", (belief_id,))
        return Belief.model_validate_json(rows[0]["data"]) if rows else None

    def delete(self, belief_id: str) -> bool:
        return self._exec("DELETE FROM beliefs WHERE id = ?", (belief_id,)) > 0

    def all(
        self,
        *,
        min_confidence: float = 0.0,
        tier: MemoryTier | None = None,
        scope: str | None = None,
        limit: int | None = None,
        include_retired: bool = False,
    ) -> list[Belief]:
        """Beliefs above a confidence floor, strongest first.

        `tier` and `scope` narrow recall to one body of memory. Filtering in SQL
        rather than in Python keeps the index doing the work.

        Superseded beliefs are left out by default — this is the one gate every
        reader passes through, so retiring a claim here retires it everywhere
        rather than in whichever call sites remembered to check. `include_retired`
        is for the callers that must see everything: deletion, and the timeline.
        """
        sql = "SELECT data FROM beliefs WHERE confidence >= ?"
        params: list[object] = [min_confidence]
        if not include_retired:
            sql += " AND superseded_by IS NULL"
        if tier is not None:
            sql += " AND tier = ?"
            params.append(str(tier))
        if scope is not None:
            sql += " AND scope = ?"
            params.append(scope)
        sql += " ORDER BY confidence DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [Belief.model_validate_json(r["data"]) for r in self._read(sql, params)]

    def fingerprints(self) -> dict[str, str]:
        """Every belief id with its last-updated stamp.

        The cheap half of `all()`: two indexed columns, no JSON and no Pydantic
        parsing. Indexes use it to work out what actually changed instead of
        rebuilding themselves, which is the difference between a new note
        costing milliseconds and costing seconds.

        Superseded beliefs are absent for the same reason they are absent from
        `all()` — an index syncing against this treats a missing id as deleted,
        which is precisely what retiring a claim should mean to retrieval.
        """
        rows = self._read("SELECT id, updated_at FROM beliefs WHERE superseded_by IS NULL")
        return {row["id"]: row["updated_at"] for row in rows}

    def count_by_tier(self) -> dict[str, int]:
        rows = self._read("SELECT tier, COUNT(*) AS n FROM beliefs GROUP BY tier ORDER BY tier")
        return {row["tier"]: row["n"] for row in rows}

    def search(self, text: str, *, limit: int = 20) -> list[Belief]:
        """Substring match on the statement, case-insensitively.

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
        return self._exec("DELETE FROM beliefs WHERE confidence <= ?", (threshold,))

    def __len__(self) -> int:
        return self._read("SELECT COUNT(*) AS n FROM beliefs")[0]["n"]


def open_store(target: str | Path = ":memory:") -> BeliefStore:
    """Open the registry named by a path or a PostgreSQL URL.

    `CME_DATABASE` flows through here, so switching backends is configuration
    rather than a code change.
    """
    text = str(target)
    if text.startswith(("postgres://", "postgresql://")):
        from cme_python.postgres_store import PostgresBeliefStore  # noqa: PLC0415

        return PostgresBeliefStore(text)
    return BeliefStore(text)
