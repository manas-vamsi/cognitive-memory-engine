"""PostgreSQL registry — the relational store the spec names for deployment.

SQLite is one writer at a time. That is fine for a single process and wrong for
several, which is the point at which this exists.

Everything is inherited from `BeliefStore`; only the genuine dialect
differences live here. `psycopg` is imported lazily and is not a CME dependency.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from cme_python.store import BeliefStore


def _missing() -> RuntimeError:
    return RuntimeError(
        "PostgresBeliefStore needs `psycopg`, which CME does not install by "
        "default. Run `pip install 'psycopg[binary]'`, or use the SQLite store."
    )


class PostgresBeliefStore(BeliefStore):
    """The belief registry on PostgreSQL.

        store = PostgresBeliefStore("postgresql://user:pass@localhost/cme")

    or set `CME_DATABASE` to the same URL and let `open_store` pick it up.
    """

    placeholder = "%s"
    like = "ILIKE"
    """Postgres LIKE is case-sensitive; SQLite's is not.

    Left as plain LIKE, the Belief Engine would stop treating "Rust is fast"
    and "rust is fast" as duplicates on Postgres alone — the same registry
    accumulating different beliefs depending on its backend. ILIKE restores
    the behaviour the rest of the engine already assumes.
    """

    def _connect(self):
        """One connection per thread, created on first use.

        The SQLite store shares one connection behind one lock, and measurement
        said the lock was not what its callers were waiting on: that work is
        CPU-bound Python under the GIL. Postgres is the opposite case. Every
        query is a network round trip and psycopg releases the GIL for its
        duration, so queries from different threads genuinely overlap — unless
        they are serialised behind one connection, which is what this used to do.

        A connection per thread rather than a pool. A pool charges a checkout on
        every query, and measured here that cost more single-threaded than the
        concurrency was worth; a thread that keeps its own connection pays it
        once. The ceiling is one backend process per calling thread, which is
        the same order as a pool and known in advance.
        """
        self._conns: dict[int, object] = {}
        conn = self._new_connection()
        self._conns[threading.get_ident()] = conn
        return conn

    def _new_connection(self):
        try:
            import psycopg  # noqa: PLC0415
            from psycopg.rows import dict_row  # noqa: PLC0415
        except ImportError:
            raise _missing() from None

        # autocommit is not a preference here, it is what makes per-thread
        # connections safe. Without it psycopg opens a transaction on the first
        # statement and holds it until someone commits — so a thread that only
        # ever reads sits `idle in transaction` forever, holding locks, and the
        # next write blocks behind it. That is not theoretical: it deadlocked
        # the test suite, and Postgres named the cause in pg_stat_activity.
        #
        # Writes still get a real transaction from `conn.transaction()`, which
        # issues its own BEGIN/COMMIT rather than relying on the implicit one.
        return psycopg.connect(self.path, row_factory=dict_row, autocommit=True)

    def _conn(self):
        """This thread's connection, opened the first time it asks."""
        key = threading.get_ident()
        conn = self._conns.get(key)
        if conn is None or conn.closed:
            conn = self._conns[key] = self._new_connection()
        return conn

    def close(self) -> None:
        with self._lock:
            for conn in self._conns.values():
                conn.close()
            self._conns.clear()

    def _existing_columns(self) -> set[str]:
        rows = self._read(
            "SELECT column_name AS name FROM information_schema.columns "
            "WHERE table_name = 'beliefs'"
        )
        return {row["name"] for row in rows}

    @contextmanager
    def _write(self) -> Iterator:
        """Commit a statement on this thread's connection.

        `with connection:` means different things in the two drivers, and the
        difference is not subtle: sqlite3 commits the transaction and leaves the
        connection open, while psycopg3 commits *and closes the connection*.
        Written the sqlite3 way, the first write here closed the connection and
        every later query failed with "the connection is closed".

        `connection.transaction()` is the psycopg3 primitive that means what
        sqlite3's `with connection:` means. No lock: the connection belongs to
        this thread alone, and Postgres arbitrates between them.
        """
        conn = self._conn()
        with conn.transaction(), conn.cursor() as cursor:
            yield cursor

    def _read(self, sql: str, params: Sequence[object] = ()) -> list[dict]:
        with self._conn().cursor() as cursor:
            cursor.execute(self._sql(sql), params)
            return cursor.fetchall()
