"""PostgreSQL registry — the relational store the spec names for deployment.

SQLite is one writer at a time. That is fine for a single process and wrong for
several, which is the point at which this exists.

Everything is inherited from `BeliefStore`; only the genuine dialect
differences live here. `psycopg` is imported lazily and is not a CME dependency.
"""

from __future__ import annotations

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
        try:
            import psycopg  # noqa: PLC0415
            from psycopg.rows import dict_row  # noqa: PLC0415
        except ImportError:
            raise _missing() from None

        return psycopg.connect(self.path, row_factory=dict_row)

    def _existing_columns(self) -> set[str]:
        rows = self._read(
            "SELECT column_name AS name FROM information_schema.columns "
            "WHERE table_name = 'beliefs'"
        )
        return {row["name"] for row in rows}

    @contextmanager
    def _write(self) -> Iterator:
        """Commit a statement.

        `with connection:` means different things in the two drivers, and the
        difference is not subtle: sqlite3 commits the transaction and leaves the
        connection open, while psycopg3 commits *and closes the connection*.
        Written the sqlite3 way, the first write here closed the connection and
        every later query failed with "the connection is closed".

        `connection.transaction()` is the psycopg3 primitive that means what
        sqlite3's `with connection:` means.
        """
        with self._lock, self._db.transaction(), self._db.cursor() as cursor:
            yield cursor

    def _read(self, sql: str, params: Sequence[object] = ()) -> list[dict]:
        with self._lock, self._db.cursor() as cursor:
            cursor.execute(self._sql(sql), params)
            return cursor.fetchall()
