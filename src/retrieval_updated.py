import math
import os
import pickle
import re
import signal
from collections import Counter, defaultdict
from difflib import get_close_matches
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from chunk_index_builder import (
    build_chunk_index_payload,
    clean_and_tokenize_text,
    get_embedding_model,
    split_text_into_sentences,
)


class BuildTimeoutError(Exception):
    pass


BUILD_TIMEOUT_SECONDS = 300


def _handle_build_timeout(signum, frame):
    raise BuildTimeoutError(
        f"Search index build exceeded {BUILD_TIMEOUT_SECONDS} seconds."
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRANSCRIPTS_PATH = os.path.join(DATA_DIR, "4300_transcripts.json")
INDEX_PATH = os.path.join(DATA_DIR, "search_index.pkl")

MIN_DF = 1
MAX_DF_RATIO = 0.15
DEFAULT_TOP_K = 50
DEFAULT_SVD_COMPONENTS = 100
SAVE_DEBUG_JSON = False
DEFAULT_MAX_CHUNKS_PER_DOC = 2
DEFAULT_RESULT_SCOPE = "chunks"

SNIPPET_WINDOW = 3
MIN_SHARED_SNIPPET_SENTENCES = 2
MIN_DISPLAY_SNIPPET_SENTENCES = 7
TARGET_DISPLAY_SNIPPET_SENTENCES = 10
MIN_DISPLAY_SNIPPET_WORDS = 90

SVD_EXPLAIN_TOP_DIMS = 3
SVD_EXPLAIN_TOP_TERMS = 8

_SEARCH_INDEX: Optional[Dict[str, Any]] = None


# -------------------------------------------------------------------
# BASIC HELPERS
# -------------------------------------------------------------------
def build_word_document_count(items: List[Dict[str, Any]]) -> Dict[str, int]:
    doc_count = Counter()
    for item in items:
        unique_tokens = set(item["tokens"])
        doc_count.update(unique_tokens)
    return dict(doc_count)


def build_good_words(
    items: List[Dict[str, Any]],
    min_df: int = MIN_DF,
    max_df_ratio: float = MAX_DF_RATIO
) -> List[str]:
    doc_count = build_word_document_count(items)
    n_docs = len(items)

    good_words = [
        word for word, df in doc_count.items()
        if df >= min_df and (df / n_docs) <= max_df_ratio
    ]
    return sorted(good_words)


def filter_tokens_to_good_words(
    items: List[Dict[str, Any]],
    good_words: List[str]
) -> List[Dict[str, Any]]:
    good_set = set(good_words)

    for item in items:
        filtered = [tok for tok in item["tokens"] if tok in good_set]
        item["tokens"] = filtered
        item["length"] = len(filtered)

    return items


def build_inverted_index(items: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, int]]]:
    index = defaultdict(list)

    for doc_id, item in enumerate(items):
        counts = Counter(item["tokens"])
        for term, tf in counts.items():
            index[term].append((doc_id, tf))

    return dict(index)


def compute_idf(
    inv_idx: Dict[str, List[Tuple[int, int]]],
    n_docs: int,
    min_df: int = MIN_DF,
    max_df_ratio: float = MAX_DF_RATIO
) -> Dict[str, float]:
    idf = {}

    for term, postings in inv_idx.items():
        df = len(postings)

        if df < min_df:
            continue
        if df / n_docs > max_df_ratio:
            continue

        idf[term] = math.log((1 + n_docs) / (1 + df)) + 1

    return idf


def create_vocab(idf: Dict[str, float]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    vocab = sorted(idf.keys())
    word_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_word = {i: w for w, i in word_to_index.items()}
    return vocab, word_to_index, index_to_word


def create_tfidf_matrix(
    items: List[Dict[str, Any]],
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
    normalize_tf: bool = False,
):
    rows = []
    cols = []
    data = []

    n_docs = len(items)
    vocab_size = len(word_to_index)

    for doc_id, item in enumerate(items):
        counts = Counter(item["tokens"])
        doc_len = len(item["tokens"])

        for term, raw_tf in counts.items():
            if term not in word_to_index:
                continue

            tf = (raw_tf / doc_len) if normalize_tf and doc_len else raw_tf
            value = tf * idf[term]

            rows.append(doc_id)
            cols.append(word_to_index[term])
            data.append(value)

    return csr_matrix((data, (rows, cols)), shape=(n_docs, vocab_size), dtype=float)


def vectorize_tokens(tokens: List[str], word_to_index: Dict[str, int], idf: Dict[str, float]) -> np.ndarray:
    counts = Counter(tokens)
    vec = np.zeros(len(word_to_index), dtype=float)

    for term, tf in counts.items():
        if term in word_to_index and term in idf:
            vec[word_to_index[term]] = tf * idf[term]

    return vec


def vectorize_query(query: str, word_to_index: Dict[str, int], idf: Dict[str, float]) -> np.ndarray:
    tokens = clean_and_tokenize_text(query)
    return vectorize_tokens(tokens, word_to_index, idf)


def cosine_scores(query_vec: np.ndarray, matrix) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=float)

    matrix_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    dots = matrix @ query_vec
    denom = matrix_norms * query_norm
    scores = np.divide(dots, denom, out=np.zeros_like(dots, dtype=float), where=denom != 0)
    return np.asarray(scores).reshape(-1)


def snippet_word_count(sentences: List[str]) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", " ".join(sentences)))


# -------------------------------------------------------------------
# SENTENCE / SNIPPET HELPERS
# -------------------------------------------------------------------
def find_best_matching_sentence_in_text(
    query: str,
    text: str,
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
) -> Tuple[Optional[int], str, List[str], float]:
    sentences = split_text_into_sentences(text)

    if not sentences:
        return None, "", [], 0.0

    query_vec = vectorize_query(query, word_to_index, idf)
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        return None, "", sentences, 0.0

    best_idx = None
    best_score = -1.0

    for i, sentence in enumerate(sentences):
        sent_tokens = clean_and_tokenize_text(sentence)
        sent_vec = vectorize_tokens(sent_tokens, word_to_index, idf)

        denom = np.linalg.norm(sent_vec) * query_norm
        if denom == 0:
            continue

        score = float(np.dot(sent_vec, query_vec) / denom)

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None, "", sentences, 0.0

    return best_idx, sentences[best_idx], sentences, float(best_score)


def build_display_snippet_from_best_sentence(
    source_sentences: List[str],
    best_global_idx: Optional[int],
) -> Tuple[str, int, int, List[str]]:
    if not source_sentences:
        return "", 0, -1, []

    if best_global_idx is None:
        best_global_idx = len(source_sentences) // 2

    start = max(0, best_global_idx - SNIPPET_WINDOW)
    end = min(len(source_sentences) - 1, best_global_idx + SNIPPET_WINDOW)

    snippet_sentences = source_sentences[start:end + 1]
    snippet_text = " ".join(snippet_sentences).strip()

    return snippet_text, start, end, snippet_sentences


def build_expanded_transcript_snippet(
    transcript_sentences: List[str],
    best_idx: Optional[int],
) -> Tuple[str, int, int, List[str]]:
    if not transcript_sentences:
        return "", 0, -1, []

    if best_idx is None:
        best_idx = len(transcript_sentences) // 2

    start = max(0, best_idx - SNIPPET_WINDOW)
    end = min(len(transcript_sentences) - 1, best_idx + SNIPPET_WINDOW)

    while True:
        snippet_sentences = transcript_sentences[start:end + 1]
        enough_sentences = len(snippet_sentences) >= MIN_DISPLAY_SNIPPET_SENTENCES
        enough_words = snippet_word_count(snippet_sentences) >= MIN_DISPLAY_SNIPPET_WORDS
        target_reached = len(snippet_sentences) >= TARGET_DISPLAY_SNIPPET_SENTENCES

        if (enough_sentences and enough_words) or target_reached:
            break

        if start == 0 and end == len(transcript_sentences) - 1:
            break

        if start > 0:
            start -= 1
        if end < len(transcript_sentences) - 1 and len(transcript_sentences[start:end + 1]) < TARGET_DISPLAY_SNIPPET_SENTENCES:
            end += 1

        if start == 0 and end == len(transcript_sentences) - 1:
            break

    snippet_sentences = transcript_sentences[start:end + 1]
    snippet_text = " ".join(snippet_sentences).strip()
    return snippet_text, start, end, snippet_sentences


def normalize_sentence_for_overlap(sentence: str) -> str:
    return " ".join(sentence.split()).strip().lower()


def snippets_overlap_by_at_least(
    snippet_a: List[str],
    snippet_b: List[str],
    min_shared: int = MIN_SHARED_SNIPPET_SENTENCES,
) -> bool:
    set_a = {normalize_sentence_for_overlap(s) for s in snippet_a if s.strip()}
    set_b = {normalize_sentence_for_overlap(s) for s in snippet_b if s.strip()}
    return len(set_a.intersection(set_b)) >= min_shared


# -------------------------------------------------------------------
# FILTER HELPERS
# -------------------------------------------------------------------
def get_known_comedians(items: List[Dict[str, Any]]) -> List[str]:
    return sorted({c["comedian"] for c in items if c.get("comedian", "").strip()})


def resolve_comedian_name(user_input: Optional[str], known_comedians: List[str]) -> Optional[str]:
    if not user_input or not user_input.strip():
        return None

    exact = [c for c in known_comedians if c.lower() == user_input.lower()]
    if exact:
        return exact[0]

    matches = get_close_matches(user_input, known_comedians, n=1, cutoff=0.7)
    return matches[0] if matches else None


def item_passes_filters(
    item: Dict[str, Any],
    resolved_comedian: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    special_type: Optional[str] = None,
    exclude_profanity: bool = False,
) -> bool:
    if resolved_comedian and item.get("comedian", "").lower() != resolved_comedian.lower():
        return False

    if special_type and item.get("special_type", "").lower() != special_type.lower():
        return False

    year_str = str(item.get("release_date", "")).strip()
    if year_min is not None or year_max is not None:
        if not year_str.isdigit():
            return False
        year = int(year_str)
        if year_min is not None and year < year_min:
            return False
        if year_max is not None and year > year_max:
            return False

    if exclude_profanity and item.get("has_profanity", False):
        return False

    return True


# -------------------------------------------------------------------
# FEATURE HELPERS
# -------------------------------------------------------------------
def compute_proximity_feature(query: str, text: str) -> float:
    query_tokens = clean_and_tokenize_text(query)
    text_tokens = clean_and_tokenize_text(text)

    if not query_tokens or not text_tokens:
        return 0.0

    positions = defaultdict(list)
    for idx, tok in enumerate(text_tokens):
        positions[tok].append(idx)

    found_positions = []
    for tok in query_tokens:
        if tok in positions:
            found_positions.extend(positions[tok])

    if len(found_positions) < 2:
        return 0.0

    found_positions.sort()
    span = found_positions[-1] - found_positions[0] + 1
    if span <= 0:
        return 0.0

    return min(1.0, len(found_positions) / span)


def compute_comedian_feature(item: Dict[str, Any], resolved_comedian: Optional[str]) -> float:
    if not resolved_comedian:
        return 0.0
    return 1.0 if item.get("comedian", "").lower() == resolved_comedian.lower() else 0.0


def combine_similarity_features(base_score: float, proximity_feature: float, comedian_feature: float) -> float:
    score = (
        0.85 * max(base_score, 0.0)
        + 0.10 * proximity_feature
        + 0.05 * comedian_feature
    )
    return max(0.0, min(1.0, score))


# -------------------------------------------------------------------
# SVD EXPLAINABILITY HELPERS
# -------------------------------------------------------------------
def build_svd_dimension_terms(
    svd_model: Optional[TruncatedSVD],
    index_to_word: Dict[int, str],
    top_terms: int = SVD_EXPLAIN_TOP_TERMS,
) -> Dict[int, Dict[str, List[str]]]:
    if svd_model is None:
        return {}

    components = svd_model.components_
    dimension_terms: Dict[int, Dict[str, List[str]]] = {}

    for dim_idx, comp in enumerate(components):
        pos_idx = np.argsort(comp)[-top_terms:][::-1]
        neg_idx = np.argsort(comp)[:top_terms]

        dimension_terms[dim_idx] = {
            "top_positive_terms": [index_to_word[i] for i in pos_idx if i in index_to_word],
            "top_negative_terms": [index_to_word[i] for i in neg_idx if i in index_to_word],
        }

    return dimension_terms


def explain_svd_match(
    q_latent: np.ndarray,
    item_latent: np.ndarray,
    dimension_terms: Dict[int, Dict[str, List[str]]],
    top_dims: int = SVD_EXPLAIN_TOP_DIMS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if q_latent is None or item_latent is None or len(q_latent) == 0 or len(item_latent) == 0:
        return [], []

    contributions = q_latent * item_latent

    positive_dims = [i for i, v in enumerate(contributions) if v > 0]
    negative_dims = [i for i, v in enumerate(contributions) if v < 0]

    positive_dims_sorted = sorted(positive_dims, key=lambda i: contributions[i], reverse=True)[:top_dims]
    negative_dims_sorted = sorted(negative_dims, key=lambda i: contributions[i])[:top_dims]

    positive = []
    for dim in positive_dims_sorted:
        term_info = dimension_terms.get(dim, {})
        positive.append({
            "dimension": dim,
            "query_weight": float(q_latent[dim]),
            "chunk_weight": float(item_latent[dim]),
            "contribution": float(contributions[dim]),
            "direction": "positive",
            "top_positive_terms": term_info.get("top_positive_terms", []),
            "top_negative_terms": term_info.get("top_negative_terms", []),
        })

    negative = []
    for dim in negative_dims_sorted:
        term_info = dimension_terms.get(dim, {})
        negative.append({
            "dimension": dim,
            "query_weight": float(q_latent[dim]),
            "chunk_weight": float(item_latent[dim]),
            "contribution": float(contributions[dim]),
            "direction": "negative",
            "top_positive_terms": term_info.get("top_positive_terms", []),
            "top_negative_terms": term_info.get("top_negative_terms", []),
        })

    return positive, negative


# -------------------------------------------------------------------
# INDEX BUILD / LOAD
# -------------------------------------------------------------------
def build_search_index() -> Dict[str, Any]:
    signal.signal(signal.SIGALRM, _handle_build_timeout)
    signal.alarm(BUILD_TIMEOUT_SECONDS)

    try:
        payload = build_chunk_index_payload(
            transcripts_path=TRANSCRIPTS_PATH,
            save_debug_json=SAVE_DEBUG_JSON,
        )
    finally:
        signal.alarm(0)

    chunks = payload["chunks"]
    docs = payload["docs"]
    transcript_chunk_ids = payload["transcript_chunk_ids"]
    chunk_embedding_matrix = payload["chunk_embedding_matrix"]

    good_words = build_good_words(chunks, min_df=MIN_DF, max_df_ratio=MAX_DF_RATIO)
    chunks = filter_tokens_to_good_words(chunks, good_words)

    inv_idx = build_inverted_index(chunks)
    idf = compute_idf(inv_idx, len(chunks), min_df=MIN_DF, max_df_ratio=MAX_DF_RATIO)
    vocab, word_to_index, index_to_word = create_vocab(idf)
    tfidf_matrix = create_tfidf_matrix(chunks, word_to_index, idf, normalize_tf=False)

    svd_model = None
    svd_doc_matrix = None
    svd_transcript_matrix = None
    dimension_terms = {}

    svd_components = min(DEFAULT_SVD_COMPONENTS, max(1, min(tfidf_matrix.shape) - 1)) if min(tfidf_matrix.shape) > 1 else 1

    if tfidf_matrix.shape[0] > 1 and tfidf_matrix.shape[1] > 1:
        svd_model = TruncatedSVD(n_components=svd_components, random_state=42)
        svd_doc_matrix = normalize(svd_model.fit_transform(tfidf_matrix))
        dimension_terms = build_svd_dimension_terms(svd_model, index_to_word)

    transcript_items = []
    for doc in docs:
        doc_id = doc["doc_id"]
        source_sentences = doc.get("sentences", split_text_into_sentences(doc.get("content", "")))
        full_text = doc.get("content", "")
        tokens = clean_and_tokenize_text(full_text)

        profanity_terms = set(doc.get("profanity_terms", []))
        has_profanity = doc.get("has_profanity", False)

        for chunk_idx in transcript_chunk_ids.get(doc_id, []):
            chunk = chunks[chunk_idx]
            if chunk.get("has_profanity", False):
                has_profanity = True
                profanity_terms.update(chunk.get("profanity_terms", []))

        item = {
            "chunk_id": f"doc_{doc_id}",
            "doc_id": doc_id,
            "comedian": doc.get("comedian", ""),
            "special_title": doc.get("special_title", ""),
            "release_date": doc.get("release_date", ""),
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "platform": doc.get("platform", ""),
            "special_type": doc.get("special_type", ""),
            "content": full_text,
            "tokens": tokens,
            "length": len(tokens),
            "chunk_sentences": source_sentences,
            "global_snippet_start": 0,
            "global_snippet_end": max(0, len(source_sentences) - 1),
            "has_profanity": has_profanity,
            "profanity_terms": sorted(profanity_terms),
        }
        transcript_items.append(item)

    transcript_tfidf_matrix = create_tfidf_matrix(transcript_items, word_to_index, idf, normalize_tf=False)

    if svd_model is not None and transcript_tfidf_matrix.shape[0] > 0:
        svd_transcript_matrix = normalize(svd_model.transform(transcript_tfidf_matrix))

    model = get_embedding_model()

    transcript_embedding_rows = []
    for doc in docs:
        text = doc.get("content", "").strip()
        if text:
            emb = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        else:
            emb = np.zeros(384, dtype=np.float32)
        transcript_embedding_rows.append(emb.astype(np.float32))

    transcript_embedding_matrix = (
        np.vstack(transcript_embedding_rows).astype(np.float32)
        if transcript_embedding_rows
        else np.zeros((0, 384), dtype=np.float32)
    )

    index = {
        "docs": docs,
        "chunks": chunks,
        "transcript_items": transcript_items,
        "transcript_chunk_ids": transcript_chunk_ids,
        "inv_idx": inv_idx,
        "idf": idf,
        "vocab": vocab,
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "tfidf_matrix": tfidf_matrix,
        "svd_model": svd_model,
        "svd_doc_matrix": svd_doc_matrix,
        "svd_transcript_matrix": svd_transcript_matrix,
        "dimension_terms": dimension_terms,
        "chunk_embedding_matrix": chunk_embedding_matrix,
        "transcript_tfidf_matrix": transcript_tfidf_matrix,
        "transcript_embedding_matrix": transcript_embedding_matrix,
    }

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    return index


def get_search_index() -> Dict[str, Any]:
    global _SEARCH_INDEX

    if _SEARCH_INDEX is not None:
        return _SEARCH_INDEX

    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f:
            _SEARCH_INDEX = pickle.load(f)
        return _SEARCH_INDEX

    _SEARCH_INDEX = build_search_index()
    return _SEARCH_INDEX


def initialize_search() -> Dict[str, Any]:
    return get_search_index()


# -------------------------------------------------------------------
# RESULT BUILDERS
# -------------------------------------------------------------------
def build_result_object(
    item: Dict[str, Any],
    docs: List[Dict[str, Any]],
    query: str,
    retrieval_mode: str,
    base_score: float,
    resolved_comedian: Optional[str],
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
    q_latent: Optional[np.ndarray] = None,
    item_latent: Optional[np.ndarray] = None,
    dimension_terms: Optional[Dict[int, Dict[str, List[str]]]] = None,
) -> Dict[str, Any]:
    is_full_transcript = str(item["chunk_id"]).startswith("doc_")
    source_sentences = item.get("chunk_sentences", [])

    if is_full_transcript:
        best_idx, best_sentence, _, sentence_score = find_best_matching_sentence_in_text(
            query=query,
            text=item["content"],
            word_to_index=word_to_index,
            idf=idf,
        )

        display_snippet, snippet_start, snippet_end, snippet_sentences = build_display_snippet_from_best_sentence(
            source_sentences=source_sentences,
            best_global_idx=best_idx,
        )

        best_sentence_index = best_idx
        global_snippet_start = snippet_start
        global_snippet_end = snippet_end
    else:
        doc = docs[item["doc_id"]]
        full_transcript_sentences = doc.get("sentences", [])
        chunk_text = item.get("content", "")
        chunk_sentence_start = item.get("sentence_start", item.get("global_snippet_start", 0))

        local_best_idx, best_sentence, _, sentence_score = find_best_matching_sentence_in_text(
            query=query,
            text=chunk_text,
            word_to_index=word_to_index,
            idf=idf,
        )

        global_best_idx = None
        if local_best_idx is not None:
            global_best_idx = chunk_sentence_start + local_best_idx

        display_snippet, snippet_start, snippet_end, snippet_sentences = build_expanded_transcript_snippet(
            transcript_sentences=full_transcript_sentences,
            best_idx=global_best_idx,
        )

        best_sentence_index = global_best_idx
        global_snippet_start = snippet_start
        global_snippet_end = snippet_end

    proximity_feature = compute_proximity_feature(query, item["content"])
    comedian_feature = compute_comedian_feature(item, resolved_comedian)
    similarity_score = combine_similarity_features(base_score, proximity_feature, comedian_feature)
    similarity_percent = similarity_score * 100.0

    svd_positive_dimensions = []
    svd_negative_dimensions = []
    if retrieval_mode == "svd" and q_latent is not None and item_latent is not None and dimension_terms is not None:
        svd_positive_dimensions, svd_negative_dimensions = explain_svd_match(
            q_latent=q_latent,
            item_latent=item_latent,
            dimension_terms=dimension_terms,
        )

    return {
        "chunk_id": item["chunk_id"],
        "doc_id": item["doc_id"],
        "comedian": item["comedian"],
        "special_title": item["special_title"],
        "release_date": item["release_date"],
        "title": item["title"],
        "url": item["url"],
        "platform": item["platform"],
        "special_type": item["special_type"],
        "content": item["content"],
        "display_snippet": display_snippet,
        "chunk_sentences": source_sentences,
        "best_sentence": best_sentence,
        "best_sentence_index": best_sentence_index,
        "sentence_score": sentence_score,
        "snippet_sentences": snippet_sentences,
        "snippet_sentence_start": snippet_start,
        "snippet_sentence_end": snippet_end,
        "global_snippet_start": global_snippet_start,
        "global_snippet_end": global_snippet_end,
        "has_profanity": item["has_profanity"],
        "profanity_terms": item.get("profanity_terms", []),
        "base_score": base_score,
        "proximity_feature": proximity_feature,
        "comedian_feature": comedian_feature,
        "similarity_score": similarity_score,
        "similarity_percent": similarity_percent,
        "retrieval_mode": retrieval_mode,
        "result_scope": "full" if is_full_transcript else "chunks",
        "svd_positive_dimensions": svd_positive_dimensions,
        "svd_negative_dimensions": svd_negative_dimensions,
    }


# -------------------------------------------------------------------
# SEARCH
# -------------------------------------------------------------------
def search_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    retrieval_mode: str = "tfidf",
    comedian: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    special_type: Optional[str] = None,
    exclude_profanity: bool = False,
    max_chunks_per_doc: int = DEFAULT_MAX_CHUNKS_PER_DOC,
    result_scope: str = DEFAULT_RESULT_SCOPE,
) -> Dict[str, Any]:
    index = get_search_index()

    docs = index["docs"]
    chunks = index["chunks"]
    transcript_items = index["transcript_items"]
    word_to_index = index["word_to_index"]
    idf = index["idf"]

    known_comedians = get_known_comedians(chunks)
    resolved_comedian = resolve_comedian_name(comedian, known_comedians)

    if result_scope not in {"chunks", "full"}:
        result_scope = "chunks"

    search_items = chunks if result_scope == "chunks" else transcript_items

    filtered_indices = [
        i for i, item in enumerate(search_items)
        if item_passes_filters(
            item=item,
            resolved_comedian=resolved_comedian,
            year_min=year_min,
            year_max=year_max,
            special_type=special_type,
            exclude_profanity=exclude_profanity,
        )
    ]

    if not filtered_indices:
        return {
            "query": query,
            "results": [],
            "resolved_comedian": resolved_comedian,
            "known_comedians": known_comedians,
            "known_special_types": sorted({c["special_type"] for c in chunks if c.get("special_type")}),
        }

    q_vec = vectorize_query(query, word_to_index, idf)
    q_latent = None
    latent_matrix = None
    dimension_terms = index.get("dimension_terms", {})

    if retrieval_mode == "tfidf":
        matrix = index["tfidf_matrix"] if result_scope == "chunks" else index["transcript_tfidf_matrix"]
        all_scores = cosine_scores(q_vec, matrix)

    elif retrieval_mode == "embedding":
        model = get_embedding_model()
        q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        matrix = index["chunk_embedding_matrix"] if result_scope == "chunks" else index["transcript_embedding_matrix"]
        all_scores = matrix @ q_emb

    elif retrieval_mode == "svd":
        svd_model = index["svd_model"]
        if svd_model is None:
            matrix = index["tfidf_matrix"] if result_scope == "chunks" else index["transcript_tfidf_matrix"]
            all_scores = cosine_scores(q_vec, matrix)
            retrieval_mode = "tfidf"
        else:
            q_latent = normalize(svd_model.transform(q_vec.reshape(1, -1)))[0]
            latent_matrix = index["svd_doc_matrix"] if result_scope == "chunks" else index["svd_transcript_matrix"]

            if latent_matrix is None:
                matrix = index["tfidf_matrix"] if result_scope == "chunks" else index["transcript_tfidf_matrix"]
                all_scores = cosine_scores(q_vec, matrix)
                retrieval_mode = "tfidf"
            else:
                all_scores = latent_matrix @ q_latent

    else:
        matrix = index["tfidf_matrix"] if result_scope == "chunks" else index["transcript_tfidf_matrix"]
        all_scores = cosine_scores(q_vec, matrix)
        retrieval_mode = "tfidf"

    filtered_scored = [(i, float(all_scores[i])) for i in filtered_indices]
    filtered_scored.sort(key=lambda x: x[1], reverse=True)

    results = []

    if result_scope == "chunks":
        per_doc_counts = defaultdict(int)
        accepted_snippets_by_doc: Dict[int, List[List[str]]] = defaultdict(list)

        for item_idx, base_score in filtered_scored:
            item = chunks[item_idx]
            doc_id = item["doc_id"]

            if per_doc_counts[doc_id] >= max_chunks_per_doc:
                continue

            item_latent = None
            if retrieval_mode == "svd" and latent_matrix is not None:
                item_latent = latent_matrix[item_idx]

            result = build_result_object(
                item=item,
                docs=docs,
                query=query,
                retrieval_mode=retrieval_mode,
                base_score=base_score,
                resolved_comedian=resolved_comedian,
                word_to_index=word_to_index,
                idf=idf,
                q_latent=q_latent,
                item_latent=item_latent,
                dimension_terms=dimension_terms,
            )

            candidate_snippet = result.get("snippet_sentences", []) or []

            has_overlap = any(
                snippets_overlap_by_at_least(candidate_snippet, accepted_snippet, MIN_SHARED_SNIPPET_SENTENCES)
                for accepted_snippet in accepted_snippets_by_doc[doc_id]
            )
            if has_overlap:
                continue

            results.append(result)
            accepted_snippets_by_doc[doc_id].append(candidate_snippet)
            per_doc_counts[doc_id] += 1

            if len(results) >= top_k:
                break

    else:
        for item_idx, base_score in filtered_scored[:top_k]:
            item = transcript_items[item_idx]

            item_latent = None
            if retrieval_mode == "svd" and latent_matrix is not None:
                item_latent = latent_matrix[item_idx]

            result = build_result_object(
                item=item,
                docs=docs,
                query=query,
                retrieval_mode=retrieval_mode,
                base_score=base_score,
                resolved_comedian=resolved_comedian,
                word_to_index=word_to_index,
                idf=idf,
                q_latent=q_latent,
                item_latent=item_latent,
                dimension_terms=dimension_terms,
            )
            results.append(result)

    return {
        "query": query,
        "results": results,
        "resolved_comedian": resolved_comedian,
        "known_comedians": known_comedians,
        "known_special_types": sorted({c["special_type"] for c in chunks if c.get("special_type")}),
    }