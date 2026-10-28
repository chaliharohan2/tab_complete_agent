"""
SQLite stand-in for the real Elasticsearch entity index.

Two tables holding the same data:
  - entities:      plain table, used for short queries (<3 chars) via LIKE
  - entities_fts:  FTS5 virtual table with the trigram tokenizer, used for
                    fuzzy substring matching on longer queries (>=3 chars)

This is intentionally kept behind a small function-based interface
(`fuzzy_search`) so swapping in real Elasticsearch later only means
rewriting this file.
"""
import sqlite3

from . import config
from .seed_data import ALL_ENTITIES


def get_connection() -> sqlite3.Connection:
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db(reset: bool = False) -> None:
    con = get_connection()
    cur = con.cursor()

    if reset:
        cur.execute("DROP TABLE IF EXISTS entities")
        cur.execute("DROP TABLE IF EXISTS entities_fts")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            type TEXT NOT NULL,
            value TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts
        USING fts5(value, type UNINDEXED, tokenize='trigram')
        """
    )

    cur.execute("SELECT COUNT(*) FROM entities")
    count = cur.fetchone()[0]
    if count == 0:
        cur.executemany(
            "INSERT INTO entities (type, value) VALUES (?, ?)",
            [(t, v) for t, v in ALL_ENTITIES],
        )
        cur.executemany(
            "INSERT INTO entities_fts (rowid, value, type) VALUES (?, ?, ?)",
            [(i + 1, v, t) for i, (t, v) in enumerate(ALL_ENTITIES)],
        )
        con.commit()

    con.close()


def _fts_escape(query: str) -> str:
    # Wrap as an explicit FTS5 phrase literal so spaces/punctuation in the
    # query aren't interpreted as FTS5 query-syntax operators.
    return '"' + query.replace('"', '""') + '"'


def fuzzy_search(query: str, types: list[str], limit_per_type: int = None) -> list[dict]:
    if not query or not types:
        return []

    limit_per_type = limit_per_type or config.RESULTS_PER_TYPE
    con = get_connection()
    results: list[dict] = []

    try:
        if len(query) >= config.MIN_TRIGRAM_QUERY_LEN:
            match = _fts_escape(query)
            for t in types:
                rows = con.execute(
                    """
                    SELECT value, type
                    FROM entities_fts
                    WHERE entities_fts MATCH ? AND type = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (match, t, limit_per_type),
                ).fetchall()
                results.extend({"value": r["value"], "type": r["type"]} for r in rows)
        else:
            # Trigram index needs >=3 chars; fall back to a plain substring LIKE.
            like_pattern = f"%{query}%"
            for t in types:
                rows = con.execute(
                    """
                    SELECT value, type
                    FROM entities
                    WHERE type = ? AND value LIKE ? COLLATE NOCASE
                    LIMIT ?
                    """,
                    (t, like_pattern, limit_per_type),
                ).fetchall()
                results.extend({"value": r["value"], "type": r["type"]} for r in rows)
    finally:
        con.close()

    return results


if __name__ == "__main__":
    init_db(reset=True)
    print("DB initialized at", config.DB_PATH)
