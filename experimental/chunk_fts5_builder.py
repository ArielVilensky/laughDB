"""
Fix #3 — SQLite FTS5 chunk index builder.

Builds a SQLite database with an FTS5 virtual table over all chunk texts.
FTS5 maintains its own C-level inverted index with BM25 ranking — queries
run in <100ms on 140K rows regardless of hardware, with near-zero RAM usage.

Usage (run once locally, then upload chunks.db to GitHub Releases):
    python experimental/chunk_fts5_builder.py

Output: src/data/chunks.db  (~25MB)
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib

CHUNK_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "data", "chunk_search_index.pkl")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "data", "chunks.db")


def build_fts5_db(index_path: str = CHUNK_INDEX_PATH, db_path: str = DB_PATH) -> None:
    print("Loading chunk index…")
    index = joblib.load(index_path)
    chunks = index["items"]
    print(f"  {len(chunks)} chunks loaded")

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    # Metadata table — holds everything except text (for filter + result building)
    conn.execute("""
        CREATE TABLE chunk_meta (
            rowid        INTEGER PRIMARY KEY,
            chunk_id     TEXT,
            doc_id       INTEGER,
            comedian     TEXT,
            special_title TEXT,
            release_date TEXT,
            special_type TEXT,
            has_profanity INTEGER,
            sentence_start INTEGER,
            sentence_end   INTEGER,
            url          TEXT,
            watch_url    TEXT,
            watch_platform TEXT
        )
    """)

    # FTS5 virtual table — content= links to chunk_meta so text isn't stored twice
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content,
            content='chunk_meta',
            content_rowid='rowid',
            tokenize='porter ascii'
        )
    """)

    print("Inserting rows…")
    meta_rows = []
    fts_rows = []

    for i, chunk in enumerate(chunks):
        content = chunk.get("content") or " ".join(chunk.get("chunk_sentences", []))
        meta_rows.append((
            i,
            chunk.get("chunk_id", f"chunk_{i}"),
            chunk.get("doc_id", 0),
            chunk.get("comedian", ""),
            chunk.get("special_title", chunk.get("title", "")),
            str(chunk.get("release_date", "")),
            chunk.get("special_type", ""),
            int(chunk.get("has_profanity", False)),
            chunk.get("sentence_start", 0),
            chunk.get("sentence_end", 0),
            chunk.get("url", ""),
            chunk.get("watch_url", ""),
            chunk.get("watch_platform", ""),
        ))
        fts_rows.append((i, content))

    conn.executemany(
        "INSERT INTO chunk_meta VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", meta_rows
    )
    conn.executemany("INSERT INTO chunks_fts(rowid, content) VALUES (?,?)", fts_rows)

    # Rebuild FTS index after bulk insert
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    conn.commit()
    conn.close()

    size_mb = os.path.getsize(db_path) / 1024 / 1024
    print(f"Done. {db_path}  ({size_mb:.1f} MB)")
    print("\nNext steps:")
    print("  1. Upload chunks.db to GitHub Releases as v4-indices/chunks.db")
    print("  2. Add wget for chunks.db in Dockerfile")
    print("  3. Replace _search_chunks_sharded with chunk_fts5_search.search_chunks_fts5")


if __name__ == "__main__":
    build_fts5_db()
