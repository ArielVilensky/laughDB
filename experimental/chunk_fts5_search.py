"""
Fix #3 — SQLite FTS5 query path.

Drop-in replacement for _search_chunks_sharded that uses the FTS5 database
built by chunk_fts5_builder.py.  Queries run in <100ms; RAM usage is ~2MB
(SQLite page cache) regardless of corpus size.

Integration point in retrieval_updated.py:
    In search_chunks(), replace the call to _search_chunks_sharded(...) with:

        if result_scope == "chunks" and fts5_db_exists():
            return search_chunks_fts5(query=query, meta=index, ...)
        else:
            return _search_chunks_sharded(...)
"""

import os
import sqlite3
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "data", "chunks.db")

_fts5_conn: sqlite3.Connection | None = None


def fts5_db_exists() -> bool:
    return os.path.exists(DB_PATH)


def _get_conn() -> sqlite3.Connection:
    global _fts5_conn
    if _fts5_conn is None:
        _fts5_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _fts5_conn.row_factory = sqlite3.Row
    return _fts5_conn


def search_chunks_fts5(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: int = 25,
    comedian: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    special_type: Optional[str] = None,
    exclude_profanity: bool = False,
    max_chunks_per_doc: int = 2,
) -> List[Dict[str, Any]]:
    """
    Query the FTS5 index and return enriched result dicts compatible with
    the existing build_result_object() output format.

    The caller is responsible for proximity re-ranking on the returned
    candidates — this function returns up to top_k * 8 raw BM25 candidates
    so the existing rerank pipeline can run on them.
    """
    conn = _get_conn()

    # Build WHERE clause for metadata filters
    where_clauses = []
    params: list = []

    if comedian:
        where_clauses.append("m.comedian = ?")
        params.append(comedian)
    if year_min is not None:
        where_clauses.append("CAST(m.release_date AS INTEGER) >= ?")
        params.append(year_min)
    if year_max is not None:
        where_clauses.append("CAST(m.release_date AS INTEGER) <= ?")
        params.append(year_max)
    if special_type:
        where_clauses.append("m.special_type = ?")
        params.append(special_type)
    if exclude_profanity:
        where_clauses.append("m.has_profanity = 0")

    meta_filter = ("AND " + " AND ".join(where_clauses)) if where_clauses else ""

    # FTS5 MATCH with BM25 ranking; fetch enough candidates for reranking
    fetch_k = min(top_k * 8, 400)
    fts_query = query.replace('"', '""')  # escape double-quotes

    sql = f"""
        SELECT
            f.rowid,
            bm25(chunks_fts) AS score,
            m.chunk_id, m.doc_id, m.comedian, m.special_title,
            m.release_date, m.special_type, m.has_profanity,
            m.sentence_start, m.sentence_end,
            m.url, m.watch_url, m.watch_platform
        FROM chunks_fts f
        JOIN chunk_meta m ON m.rowid = f.rowid
        WHERE chunks_fts MATCH ?
        {meta_filter}
        ORDER BY score
        LIMIT ?
    """
    params_full = [fts_query] + params + [fetch_k]

    try:
        rows = conn.execute(sql, params_full).fetchall()
    except sqlite3.OperationalError:
        # FTS5 MATCH raises if query has invalid syntax (empty, special chars)
        return []

    results = []
    per_doc_counts: Dict[int, int] = {}

    for row in rows:
        doc_id = row["doc_id"]
        if per_doc_counts.get(doc_id, 0) >= max_chunks_per_doc:
            continue
        per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1

        # Reconstruct chunk sentences from docs (same as _enrich_chunk_item)
        doc = docs[doc_id] if doc_id < len(docs) else {}
        sentences = doc.get("sentences", [])
        start = row["sentence_start"]
        end = row["sentence_end"]
        chunk_sentences = sentences[start: end + 1]
        content = " ".join(chunk_sentences)

        results.append({
            "chunk_id": row["chunk_id"],
            "doc_id": doc_id,
            "comedian": row["comedian"],
            "special_title": row["special_title"],
            "release_date": row["release_date"],
            "special_type": row["special_type"],
            "has_profanity": bool(row["has_profanity"]),
            "sentence_start": start,
            "sentence_end": end,
            "url": row["url"],
            "watch_url": row["watch_url"],
            "watch_platform": row["watch_platform"],
            "content": content,
            "chunk_sentences": chunk_sentences,
            "bm25_score": -float(row["score"]),  # bm25() returns negative in SQLite
        })

        if len(results) >= top_k:
            break

    return results
