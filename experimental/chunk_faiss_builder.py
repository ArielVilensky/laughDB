"""
Fix #4 — FAISS flat index builder.

Builds a FAISS IndexFlatIP (inner product = cosine on normalized vectors) over
all 140K chunk SVD vectors.  FAISS runs in C++ with BLAS — searching 140K
float32 vectors in 130 dimensions takes ~2ms vs ~2s with numpy on server CPUs.

Prerequisites:
    pip install faiss-cpu

Usage (run once locally, then upload chunks_faiss.index to GitHub Releases):
    python experimental/chunk_faiss_builder.py

Output:
    src/data/chunks_faiss.index  (~75MB, binary)
    src/data/chunks_faiss_meta.pkl  (rowid → chunk metadata, ~20MB)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import joblib

CHUNK_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "data", "chunk_search_index.pkl")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "data", "chunks_faiss.index")
FAISS_META_PATH  = os.path.join(os.path.dirname(__file__), "..", "src", "data", "chunks_faiss_meta.pkl")


def build_faiss_index(index_path: str = CHUNK_INDEX_PATH) -> None:
    try:
        import faiss
    except ImportError:
        print("faiss-cpu not installed.  Run: pip install faiss-cpu")
        return

    print("Loading chunk index…")
    index = joblib.load(index_path)

    svd_matrix = index.get("svd_matrix")
    if svd_matrix is None:
        print("No SVD matrix in index — build with SVD enabled first.")
        return

    chunks = index["items"]
    print(f"  {len(chunks)} chunks, SVD matrix shape: {svd_matrix.shape}")

    # Normalize rows so IndexFlatIP == cosine similarity
    vectors = svd_matrix.astype(np.float32)
    faiss.normalize_L2(vectors)

    d = vectors.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(vectors)

    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    size_mb = os.path.getsize(FAISS_INDEX_PATH) / 1024 / 1024
    print(f"Saved FAISS index → {FAISS_INDEX_PATH}  ({size_mb:.1f} MB)")

    # Save lightweight metadata (no text fields) for result building
    meta = [
        {
            "chunk_id": c.get("chunk_id", f"chunk_{i}"),
            "doc_id": c.get("doc_id", 0),
            "comedian": c.get("comedian", ""),
            "special_title": c.get("special_title", c.get("title", "")),
            "release_date": c.get("release_date", ""),
            "special_type": c.get("special_type", ""),
            "has_profanity": c.get("has_profanity", False),
            "sentence_start": c.get("sentence_start", 0),
            "sentence_end": c.get("sentence_end", 0),
            "url": c.get("url", ""),
            "watch_url": c.get("watch_url", ""),
            "watch_platform": c.get("watch_platform", ""),
        }
        for i, c in enumerate(chunks)
    ]
    joblib.dump(meta, FAISS_META_PATH, compress=3)
    meta_mb = os.path.getsize(FAISS_META_PATH) / 1024 / 1024
    print(f"Saved FAISS meta  → {FAISS_META_PATH}  ({meta_mb:.1f} MB)")

    print("\nNext steps:")
    print("  1. Upload chunks_faiss.index + chunks_faiss_meta.pkl to GitHub Releases (v4-indices)")
    print("  2. Add wget lines in Dockerfile")
    print("  3. Replace SVD scoring in _search_chunks_sharded with chunk_faiss_search.search_chunks_faiss")


if __name__ == "__main__":
    build_faiss_index()
