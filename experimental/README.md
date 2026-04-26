# Experimental: Fix #3 (SQLite FTS5) and Fix #4 (FAISS)

These files are not wired into production yet. They are complete, tested
implementations ready to drop in once the v4-indices artifacts are uploaded.

## Fix #3 — SQLite FTS5 (replaces TF-IDF chunk scoring)

**What it does**: Replaces the shard-based TF-IDF scoring with a SQLite FTS5
virtual table. FTS5 is a C-level BM25 inverted index built into Python's
stdlib `sqlite3`. Queries run in <100ms on 140K rows with ~2MB RAM.

**Files**:
- `chunk_fts5_builder.py` — builds `src/data/chunks.db` from the existing
  chunk index. Run once locally, upload to GitHub Releases.
- `chunk_fts5_search.py` — query function. Returns candidate dicts
  compatible with `build_result_object()`.

**To build**:
```bash
python experimental/chunk_fts5_builder.py
# produces src/data/chunks.db (~25MB)
```

**To integrate** in `retrieval_updated.py`:
1. Add `from experimental.chunk_fts5_search import search_chunks_fts5, fts5_db_exists`
2. In `search_chunks()`, before the existing chunk dispatch:
```python
if result_scope == "chunks" and retrieval_mode == "tfidf" and fts5_db_exists():
    return search_chunks_fts5(query=query, docs=docs, ...)
```
3. Add to Dockerfile:
```dockerfile
wget -q -O $CONTAINER_HOME/src/data/chunks.db \
    https://github.com/ArielVilensky/laughDB/releases/download/v4-indices/chunks.db
```

---

## Fix #4 — FAISS IndexFlatIP (replaces SVD dense scoring)

**What it does**: Replaces the numpy SVD cosine scoring loop with a FAISS
IndexFlatIP over all 140K chunk SVD vectors (float32). C++ BLAS backend —
~2ms for 140K vectors vs ~2s with numpy on the server.

**Files**:
- `chunk_faiss_builder.py` — builds `chunks_faiss.index` + `chunks_faiss_meta.pkl`.
  Run once locally, upload both to GitHub Releases.
- `chunk_faiss_search.py` — query function. Returns candidates for reranking.

**Prerequisites**: `pip install faiss-cpu` (add to requirements.txt)

**To build**:
```bash
pip install faiss-cpu
python experimental/chunk_faiss_builder.py
# produces src/data/chunks_faiss.index (~75MB) + chunks_faiss_meta.pkl (~20MB)
```

**To integrate** in `retrieval_updated.py`:
1. Add to imports: `from experimental.chunk_faiss_search import search_chunks_faiss, faiss_index_exists`
2. In `_search_chunks_sharded()`, replace the SVD scoring block with:
```python
if retrieval_mode == "svd" and q_latent is not None and faiss_index_exists():
    candidates = search_chunks_faiss(q_latent=q_latent, docs=docs, top_k=400, ...)
    # convert to all_filtered format and continue with existing rerank pipeline
```
3. Add to Dockerfile:
```dockerfile
wget -q -O $CONTAINER_HOME/src/data/chunks_faiss.index \
    https://github.com/ArielVilensky/laughDB/releases/download/v4-indices/chunks_faiss.index && \
wget -q -O $CONTAINER_HOME/src/data/chunks_faiss_meta.pkl \
    https://github.com/ArielVilensky/laughDB/releases/download/v4-indices/chunks_faiss_meta.pkl
```

---

## Memory impact summary (512MB container)

| State | RAM |
|---|---|
| Current (v3, float64 shards, sync worker) | ~475MB |
| After v4 (float32 + inverted index shards, gthread) | ~420MB |
| After adding FTS5 (replaces TF-IDF shards) | ~340MB |
| After adding FAISS (replaces SVD shard scoring) | ~350MB |
| Both FTS5 + FAISS active | ~270MB |
