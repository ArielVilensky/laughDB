"""
Fix #4 — FAISS query path for SVD/dense vector chunk search.

Drop-in replacement for the SVD branch of _search_chunks_sharded.
Searching 140K float32 vectors in 130 dims takes ~2ms with FAISS vs ~2s
with numpy on the course server.

Integration point in retrieval_updated.py:
    In _search_chunks_sharded(), replace the SVD scoring block with:

        if retrieval_mode == "svd" and faiss_index_exists():
            return search_chunks_faiss(q_latent=q_latent, top_k=top_k * 8, ...)
        # else fall through to existing shard scoring
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np

FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "data", "chunks_faiss.index")
FAISS_META_PATH  = os.path.join(os.path.dirname(__file__), "..", "src", "data", "chunks_faiss_meta.pkl")

_faiss_index = None
_faiss_meta: list | None = None


def faiss_index_exists() -> bool:
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH)


def _load_faiss():
    global _faiss_index, _faiss_meta
    if _faiss_index is None:
        import faiss
        import joblib
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        _faiss_meta = joblib.load(FAISS_META_PATH)


def search_chunks_faiss(
    q_latent: np.ndarray,
    docs: List[Dict[str, Any]],
    top_k: int = 200,
    comedian: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    special_type: Optional[str] = None,
    exclude_profanity: bool = False,
    max_chunks_per_doc: int = 2,
) -> List[Dict[str, Any]]:
    """
    Search using FAISS inner product over normalized SVD vectors.
    Returns enriched candidate dicts for the existing rerank pipeline.

    Note: FAISS doesn't support metadata filtering pre-search, so we fetch
    top_k * 4 candidates then filter.  With 140K items this is still ~2ms.
    """
    import faiss

    _load_faiss()

    query = np.array(q_latent, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(query)

    fetch_k = min(top_k * 4, len(_faiss_meta))
    scores, indices = _faiss_index.search(query, fetch_k)
    scores = scores[0]
    indices = indices[0]

    results = []
    per_doc_counts: Dict[int, int] = {}

    for score, idx in zip(scores.tolist(), indices.tolist()):
        if idx < 0:
            continue
        meta = _faiss_meta[idx]

        # Metadata filters
        if comedian and meta["comedian"] != comedian:
            continue
        if year_min is not None:
            try:
                if int(str(meta["release_date"]).strip()) < year_min:
                    continue
            except (ValueError, TypeError):
                pass
        if year_max is not None:
            try:
                if int(str(meta["release_date"]).strip()) > year_max:
                    continue
            except (ValueError, TypeError):
                pass
        if special_type and meta["special_type"] != special_type:
            continue
        if exclude_profanity and meta["has_profanity"]:
            continue

        doc_id = meta["doc_id"]
        if per_doc_counts.get(doc_id, 0) >= max_chunks_per_doc:
            continue
        per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1

        doc = docs[doc_id] if doc_id < len(docs) else {}
        sentences = doc.get("sentences", [])
        start = meta["sentence_start"]
        end = meta["sentence_end"]
        chunk_sentences = sentences[start: end + 1]
        content = " ".join(chunk_sentences)

        results.append({
            **meta,
            "content": content,
            "chunk_sentences": chunk_sentences,
            "faiss_score": float(score),
            "global_idx": idx,
        })

        if len(results) >= top_k:
            break

    return results
