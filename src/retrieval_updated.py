import math
import os
import pickle
import re
from collections import Counter, defaultdict
from difflib import get_close_matches
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from index_builder import (
    build_transcript_index_payload,
    clean_and_tokenize_text,
)
from chunk_index_builder import build_semantic_chunks


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRANSCRIPTS_PATH = os.path.join(DATA_DIR, "4300_transcripts.json")

TRANSCRIPT_DOCS_PATH = os.path.join(DATA_DIR, "transcript_docs.pkl")
TRANSCRIPT_INDEX_PATH = os.path.join(DATA_DIR, "transcript_search_index.pkl")
CHUNK_INDEX_PATH = os.path.join(DATA_DIR, "chunk_search_index.pkl")

MIN_DF = 1
TRANSCRIPT_MAX_DF_RATIO = 0.95
CHUNK_MAX_DF_RATIO = 0.20

DEFAULT_TOP_K = 25
DEFAULT_SVD_COMPONENTS = 100
DEFAULT_MAX_CHUNKS_PER_DOC = 2
DEFAULT_RESULT_SCOPE = "full"

SNIPPET_WINDOW = 2
MIN_SHARED_SNIPPET_SENTENCES = 2
MIN_DISPLAY_SNIPPET_SENTENCES = 5
TARGET_DISPLAY_SNIPPET_SENTENCES = 7
MIN_DISPLAY_SNIPPET_WORDS = 85

SVD_EXPLAIN_TOP_DIMS = 3
SVD_EXPLAIN_TOP_TERMS = 8

_SEARCH_INDEX: Dict[str, Optional[Dict[str, Any]]] = {
    "transcript": None,
    "chunk": None,
}


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def save_pickle(path: str, obj: Any) -> None:
    ensure_data_dir()
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_word_document_count(items: List[Dict[str, Any]]) -> Dict[str, int]:
    doc_count = Counter()
    for item in items:
        unique_tokens = set(item["tokens"])
        doc_count.update(unique_tokens)
    return dict(doc_count)


def build_good_words(
    items: List[Dict[str, Any]],
    min_df: int = MIN_DF,
    max_df_ratio: float = 1.0,
) -> List[str]:
    doc_count = build_word_document_count(items)
    n_docs = max(1, len(items))

    return sorted([
        word for word, df in doc_count.items()
        if df >= min_df and (df / n_docs) <= max_df_ratio
    ])


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
    max_df_ratio: float = 1.0,
) -> Dict[str, float]:
    idf = {}

    for term, postings in inv_idx.items():
        df = len(postings)

        if df < min_df:
            continue
        if df / max(1, n_docs) > max_df_ratio:
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
    normalize_tf: bool = True,
):
    rows = []
    cols = []
    data = []

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

    return csr_matrix((data, (rows, cols)), shape=(len(items), vocab_size), dtype=float)


def vectorize_tokens(
    tokens: List[str],
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
    normalize_tf: bool = True,
) -> np.ndarray:
    counts = Counter(tokens)
    vec = np.zeros(len(word_to_index), dtype=float)
    token_len = len(tokens)

    for term, raw_tf in counts.items():
        if term in word_to_index and term in idf:
            tf = (raw_tf / token_len) if normalize_tf and token_len else raw_tf
            vec[word_to_index[term]] = tf * idf[term]

    return vec


def vectorize_query(
    query: str,
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
) -> np.ndarray:
    tokens = clean_and_tokenize_text(query)
    return vectorize_tokens(tokens, word_to_index, idf, normalize_tf=True)


def cosine_scores_dense(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0 or matrix.size == 0:
        return np.zeros(matrix.shape[0], dtype=float)

    matrix_norms = np.linalg.norm(matrix, axis=1)
    dots = matrix @ query_vec
    denom = matrix_norms * query_norm

    return np.divide(dots, denom, out=np.zeros_like(dots, dtype=float), where=denom != 0)


def cosine_scores_sparse(query_vec: np.ndarray, matrix) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=float)

    matrix_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    dots = matrix @ query_vec
    denom = matrix_norms * query_norm

    return np.divide(dots, denom, out=np.zeros_like(dots, dtype=float), where=denom != 0)


def snippet_word_count(sentences: List[str]) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", " ".join(sentences)))


def find_best_matching_sentence_from_tokens(
    query: str,
    sentences: List[str],
    sentence_tokens: List[List[str]],
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
) -> Tuple[Optional[int], str, float]:
    if not sentences or not sentence_tokens:
        return None, "", 0.0

    query_vec = vectorize_query(query, word_to_index, idf)
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        return None, "", 0.0

    best_idx = None
    best_score = -1.0

    for i, tokens in enumerate(sentence_tokens):
        if not tokens:
            continue

        sent_vec = vectorize_tokens(tokens, word_to_index, idf, normalize_tf=True)
        sent_norm = np.linalg.norm(sent_vec)

        if sent_norm == 0:
            continue

        score = float(np.dot(query_vec, sent_vec) / (query_norm * sent_norm))
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None, "", 0.0

    return best_idx, sentences[best_idx], best_score


def build_display_snippet_from_best_sentence(
    source_sentences: List[str],
    best_global_idx: Optional[int],
) -> Tuple[str, int, int, List[str]]:
    if not source_sentences:
        return "", 0, 0, []

    if best_global_idx is None:
        start = 0
        end = min(len(source_sentences), TARGET_DISPLAY_SNIPPET_SENTENCES)
        snippet_sentences = source_sentences[start:end]
        return " ".join(snippet_sentences), start, end, snippet_sentences

    start = max(0, best_global_idx - SNIPPET_WINDOW)
    end = min(len(source_sentences), best_global_idx + SNIPPET_WINDOW + 1)
    snippet_sentences = source_sentences[start:end]

    while (
        (len(snippet_sentences) < MIN_DISPLAY_SNIPPET_SENTENCES or snippet_word_count(snippet_sentences) < MIN_DISPLAY_SNIPPET_WORDS)
        and (start > 0 or end < len(source_sentences))
    ):
        expanded = False

        if start > 0:
            start -= 1
            expanded = True
        if end < len(source_sentences):
            end += 1
            expanded = True

        snippet_sentences = source_sentences[start:end]

        if not expanded:
            break

    if len(snippet_sentences) > TARGET_DISPLAY_SNIPPET_SENTENCES and best_global_idx is not None:
        half_window = TARGET_DISPLAY_SNIPPET_SENTENCES // 2
        new_start = max(0, best_global_idx - half_window)
        new_end = min(len(source_sentences), new_start + TARGET_DISPLAY_SNIPPET_SENTENCES)
        if new_end - new_start < TARGET_DISPLAY_SNIPPET_SENTENCES:
            new_start = max(0, new_end - TARGET_DISPLAY_SNIPPET_SENTENCES)

        trimmed = source_sentences[new_start:new_end]
        if best_global_idx >= new_start and best_global_idx < new_end and snippet_word_count(trimmed) >= MIN_DISPLAY_SNIPPET_WORDS:
            start, end, snippet_sentences = new_start, new_end, trimmed

    return " ".join(snippet_sentences), start, end, snippet_sentences


def snippets_overlap_enough(a: List[str], b: List[str], min_shared: int = MIN_SHARED_SNIPPET_SENTENCES) -> bool:
    if not a or not b:
        return False
    set_a = set(s.strip() for s in a if s.strip())
    set_b = set(s.strip() for s in b if s.strip())
    return len(set_a.intersection(set_b)) >= min_shared


def compute_proximity_feature(query: str, content: str) -> float:
    query_tokens = clean_and_tokenize_text(query)
    content_tokens = clean_and_tokenize_text(content)

    if not query_tokens or not content_tokens:
        return 0.0

    positions = defaultdict(list)
    for idx, tok in enumerate(content_tokens):
        positions[tok].append(idx)

    found_positions = []
    for qt in query_tokens:
        if qt in positions:
            found_positions.extend(positions[qt][:3])

    if len(found_positions) < 2:
        return 0.0

    found_positions = sorted(found_positions)
    spread = found_positions[-1] - found_positions[0] + 1
    if spread <= 0:
        return 0.0

    return min(1.0, len(found_positions) / spread)


def compute_comedian_feature(item: Dict[str, Any], resolved_comedian: Optional[str]) -> float:
    if not resolved_comedian:
        return 0.0
    return 1.0 if item.get("comedian", "").strip().lower() == resolved_comedian.strip().lower() else 0.0


def combine_similarity_features(base_score: float, proximity_feature: float, comedian_feature: float) -> float:
    score = 0.85 * base_score + 0.10 * proximity_feature + 0.05 * comedian_feature
    return max(0.0, min(1.0, score))


def get_known_comedians(items: List[Dict[str, Any]]) -> List[str]:
    return sorted({item["comedian"] for item in items if item.get("comedian")})


def resolve_comedian_name(name: Optional[str], known_comedians: List[str]) -> Optional[str]:
    if not name:
        return None

    stripped = name.strip()
    if not stripped:
        return None

    exact = [c for c in known_comedians if c.lower() == stripped.lower()]
    if exact:
        return exact[0]

    matches = get_close_matches(stripped, known_comedians, n=1, cutoff=0.75)
    return matches[0] if matches else stripped


def item_passes_filters(
    item: Dict[str, Any],
    resolved_comedian: Optional[str],
    year_min: Optional[int],
    year_max: Optional[int],
    special_type: Optional[str],
    exclude_profanity: bool,
) -> bool:
    if resolved_comedian:
        if item.get("comedian", "").strip().lower() != resolved_comedian.strip().lower():
            return False

    if special_type:
        if item.get("special_type", "") != special_type:
            return False

    release_date = item.get("release_date", "")
    if release_date and release_date.isdigit():
        year = int(release_date)
        if year_min is not None and year < year_min:
            return False
        if year_max is not None and year > year_max:
            return False

    if exclude_profanity and item.get("has_profanity", False):
        return False

    return True


def build_svd_dimension_terms(
    svd_model: TruncatedSVD,
    index_to_word: Dict[int, str],
    top_terms: int = SVD_EXPLAIN_TOP_TERMS,
) -> Dict[int, Dict[str, List[str]]]:
    dimension_terms: Dict[int, Dict[str, List[str]]] = {}

    for dim_idx, component in enumerate(svd_model.components_):
        ranked = np.argsort(component)
        top_negative = [index_to_word[i] for i in ranked[:top_terms]]
        top_positive = [index_to_word[i] for i in ranked[::-1][:top_terms]]

        dimension_terms[dim_idx] = {
            "top_positive_terms": top_positive,
            "top_negative_terms": top_negative,
        }

    return dimension_terms


def explain_svd_match(
    q_latent: np.ndarray,
    item_latent: np.ndarray,
    dimension_terms: Dict[int, Dict[str, List[str]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    contributions = q_latent * item_latent

    positive_dims_sorted = [
        i for i in np.argsort(contributions)[::-1]
        if contributions[i] > 0
    ][:SVD_EXPLAIN_TOP_DIMS]

    negative_dims_sorted = [
        i for i in np.argsort(contributions)
        if contributions[i] < 0
    ][:SVD_EXPLAIN_TOP_DIMS]

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


def build_transcript_docs() -> Dict[str, Any]:
    print("Building cleaned transcript docs...")
    payload = build_transcript_index_payload(TRANSCRIPTS_PATH)
    save_pickle(TRANSCRIPT_DOCS_PATH, payload)
    print("Saved cleaned transcript docs.")
    return payload


def load_or_build_transcript_docs() -> Dict[str, Any]:
    if os.path.exists(TRANSCRIPT_DOCS_PATH):
        print("Loading cleaned transcript docs...")
        return load_pickle(TRANSCRIPT_DOCS_PATH)
    return build_transcript_docs()


def build_transcript_search_index_from_docs(docs_payload: Dict[str, Any]) -> Dict[str, Any]:
    docs = docs_payload["docs"]

    transcript_items = []
    for doc in docs:
        transcript_items.append({
            "chunk_id": f"doc_{doc['doc_id']}",
            "doc_id": doc["doc_id"],
            "comedian": doc.get("comedian", ""),
            "special_title": doc.get("special_title", ""),
            "release_date": doc.get("release_date", ""),
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "platform": doc.get("platform", ""),
            "special_type": doc.get("special_type", ""),
            "content": doc.get("content", ""),
            "tokens": list(doc.get("tokens", [])),
            "length": len(doc.get("tokens", [])),
            "chunk_sentences": doc.get("sentences", []),
            "chunk_sentence_tokens": doc.get("sentence_tokens", []),
            "sentence_start": 0,
            "sentence_end": max(0, len(doc.get("sentences", [])) - 1),
            "global_snippet_start": 0,
            "global_snippet_end": max(0, len(doc.get("sentences", [])) - 1),
            "has_profanity": doc.get("has_profanity", False),
            "profanity_terms": list(doc.get("profanity_terms", [])),
        })

    transcript_good_words = build_good_words(
        transcript_items,
        min_df=MIN_DF,
        max_df_ratio=TRANSCRIPT_MAX_DF_RATIO,
    )
    transcript_items = filter_tokens_to_good_words(transcript_items, transcript_good_words)

    transcript_inv_idx = build_inverted_index(transcript_items)
    transcript_idf = compute_idf(
        transcript_inv_idx,
        len(transcript_items),
        min_df=MIN_DF,
        max_df_ratio=TRANSCRIPT_MAX_DF_RATIO,
    )

    transcript_vocab, transcript_word_to_index, transcript_index_to_word = create_vocab(transcript_idf)
    transcript_tfidf_matrix = create_tfidf_matrix(
        transcript_items,
        transcript_word_to_index,
        transcript_idf,
        normalize_tf=True,
    )

    transcript_svd_model = None
    transcript_svd_matrix = None
    transcript_dimension_terms = {}

    if transcript_tfidf_matrix.shape[0] > 1 and transcript_tfidf_matrix.shape[1] > 1:
        svd_components = min(
            DEFAULT_SVD_COMPONENTS,
            max(1, min(transcript_tfidf_matrix.shape[0] - 1, transcript_tfidf_matrix.shape[1] - 1))
        )
        transcript_svd_model = TruncatedSVD(n_components=svd_components, random_state=42)
        transcript_svd_matrix = normalize(transcript_svd_model.fit_transform(transcript_tfidf_matrix))
        transcript_dimension_terms = build_svd_dimension_terms(
            transcript_svd_model,
            transcript_index_to_word,
            top_terms=SVD_EXPLAIN_TOP_TERMS,
        )

    return {
        "docs": docs,
        "items": transcript_items,
        "idf": transcript_idf,
        "vocab": transcript_vocab,
        "word_to_index": transcript_word_to_index,
        "index_to_word": transcript_index_to_word,
        "tfidf_matrix": transcript_tfidf_matrix,
        "svd_model": transcript_svd_model,
        "svd_matrix": transcript_svd_matrix,
        "dimension_terms": transcript_dimension_terms,
    }


def build_transcript_search_index() -> Dict[str, Any]:
    docs_payload = load_or_build_transcript_docs()
    index = build_transcript_search_index_from_docs(docs_payload)
    save_pickle(TRANSCRIPT_INDEX_PATH, index)
    print("Saved transcript search index.")
    return index


def load_or_build_transcript_search_index() -> Dict[str, Any]:
    if os.path.exists(TRANSCRIPT_INDEX_PATH):
        print("Loading transcript search index...")
        return load_pickle(TRANSCRIPT_INDEX_PATH)
    return build_transcript_search_index()


def build_chunk_search_index() -> Dict[str, Any]:
    docs_payload = load_or_build_transcript_docs()
    docs = docs_payload["docs"]

    chunks, _adjacent_similarities, transcript_chunk_ids = build_semantic_chunks(docs)

    chunk_good_words = build_good_words(
        chunks,
        min_df=MIN_DF,
        max_df_ratio=CHUNK_MAX_DF_RATIO,
    )
    chunks = filter_tokens_to_good_words(chunks, chunk_good_words)

    chunk_inv_idx = build_inverted_index(chunks)
    chunk_idf = compute_idf(
        chunk_inv_idx,
        len(chunks),
        min_df=MIN_DF,
        max_df_ratio=CHUNK_MAX_DF_RATIO,
    )

    chunk_vocab, chunk_word_to_index, chunk_index_to_word = create_vocab(chunk_idf)
    chunk_tfidf_matrix = create_tfidf_matrix(
        chunks,
        chunk_word_to_index,
        chunk_idf,
        normalize_tf=True,
    )

    chunk_svd_model = None
    chunk_svd_matrix = None
    chunk_dimension_terms = {}

    if chunk_tfidf_matrix.shape[0] > 1 and chunk_tfidf_matrix.shape[1] > 1:
        svd_components = min(
            DEFAULT_SVD_COMPONENTS,
            max(1, min(chunk_tfidf_matrix.shape[0] - 1, chunk_tfidf_matrix.shape[1] - 1))
        )
        chunk_svd_model = TruncatedSVD(n_components=svd_components, random_state=42)
        chunk_svd_matrix = normalize(chunk_svd_model.fit_transform(chunk_tfidf_matrix))
        chunk_dimension_terms = build_svd_dimension_terms(
            chunk_svd_model,
            chunk_index_to_word,
            top_terms=SVD_EXPLAIN_TOP_TERMS,
        )

    index = {
        "docs": docs,
        "items": chunks,
        "transcript_chunk_ids": transcript_chunk_ids,
        "idf": chunk_idf,
        "vocab": chunk_vocab,
        "word_to_index": chunk_word_to_index,
        "index_to_word": chunk_index_to_word,
        "tfidf_matrix": chunk_tfidf_matrix,
        "svd_model": chunk_svd_model,
        "svd_matrix": chunk_svd_matrix,
        "dimension_terms": chunk_dimension_terms,
    }
    save_pickle(CHUNK_INDEX_PATH, index)
    print("Saved chunk search index.")
    return index


def load_or_build_chunk_search_index() -> Dict[str, Any]:
    if os.path.exists(CHUNK_INDEX_PATH):
        print("Loading chunk search index...")
        return load_pickle(CHUNK_INDEX_PATH)
    return build_chunk_search_index()


def initialize_search() -> None:
    global _SEARCH_INDEX
    print("Initializing transcript search index at startup...")
    _SEARCH_INDEX["transcript"] = load_or_build_transcript_search_index()
    _SEARCH_INDEX["chunk"] = None
    print("Transcript search index ready.")


def get_transcript_index() -> Dict[str, Any]:
    global _SEARCH_INDEX
    if _SEARCH_INDEX["transcript"] is None:
        _SEARCH_INDEX["transcript"] = load_or_build_transcript_search_index()
    return _SEARCH_INDEX["transcript"]


def get_chunk_index() -> Dict[str, Any]:
    global _SEARCH_INDEX
    if _SEARCH_INDEX["chunk"] is None:
        _SEARCH_INDEX["chunk"] = load_or_build_chunk_search_index()
    return _SEARCH_INDEX["chunk"]


def build_result_object(
    *,
    query: str,
    item: Dict[str, Any],
    docs: List[Dict[str, Any]],
    base_score: float,
    retrieval_mode: str,
    resolved_comedian: Optional[str],
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
    is_full_transcript: bool,
    q_latent: Optional[np.ndarray] = None,
    item_latent: Optional[np.ndarray] = None,
    dimension_terms: Optional[Dict[int, Dict[str, List[str]]]] = None,
) -> Dict[str, Any]:
    if is_full_transcript:
        doc = docs[item["doc_id"]]
        source_sentences = doc.get("sentences", [])
        source_sentence_tokens = doc.get("sentence_tokens", [])

        best_idx, best_sentence, sentence_score = find_best_matching_sentence_from_tokens(
            query=query,
            sentences=source_sentences,
            sentence_tokens=source_sentence_tokens,
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
        chunk_sentences = item.get("chunk_sentences", [])
        chunk_sentence_tokens = item.get("chunk_sentence_tokens", [])
        chunk_sentence_start = item.get("sentence_start", item.get("global_snippet_start", 0))

        local_best_idx, best_sentence, sentence_score = find_best_matching_sentence_from_tokens(
            query=query,
            sentences=chunk_sentences,
            sentence_tokens=chunk_sentence_tokens,
            word_to_index=word_to_index,
            idf=idf,
        )

        if local_best_idx is None:
            global_best_idx = chunk_sentence_start
        else:
            global_best_idx = chunk_sentence_start + local_best_idx

        display_snippet, snippet_start, snippet_end, snippet_sentences = build_display_snippet_from_best_sentence(
            source_sentences=full_transcript_sentences,
            best_global_idx=global_best_idx,
        )

        source_sentences = chunk_sentences
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
        "watch_url": "",
        "watch_platform": "",
    }


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
    if result_scope not in {"chunks", "full"}:
        result_scope = "full"

    if retrieval_mode not in {"tfidf", "svd"}:
        retrieval_mode = "tfidf"

    index = get_transcript_index() if result_scope == "full" else get_chunk_index()

    items = index["items"]
    docs = index["docs"]
    tfidf_matrix = index["tfidf_matrix"]
    svd_model = index["svd_model"]
    svd_matrix = index["svd_matrix"]
    dimension_terms = index["dimension_terms"]
    word_to_index = index["word_to_index"]
    idf = index["idf"]

    known_comedians = get_known_comedians(items)
    resolved_comedian = resolve_comedian_name(comedian, known_comedians)

    filtered_indices = [
        i for i, item in enumerate(items)
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
            "known_special_types": sorted({item.get("special_type", "") for item in items if item.get("special_type", "")}),
        }

    query_vec = vectorize_query(query, word_to_index, idf)

    if retrieval_mode == "svd" and svd_model is not None and svd_matrix is not None:
        q_latent = svd_model.transform(query_vec.reshape(1, -1))[0]
        q_latent = normalize(q_latent.reshape(1, -1))[0]
        all_scores = cosine_scores_dense(q_latent, svd_matrix)
    else:
        q_latent = None
        all_scores = cosine_scores_sparse(query_vec, tfidf_matrix)

    scored = [(idx, float(all_scores[idx])) for idx in filtered_indices if float(all_scores[idx]) > 0]
    scored.sort(key=lambda x: x[1], reverse=True)

    if result_scope == "chunks":
        limited_scored: List[Tuple[int, float]] = []
        per_doc_counts: Dict[int, int] = defaultdict(int)

        for idx, score in scored:
            doc_id = items[idx]["doc_id"]
            if per_doc_counts[doc_id] >= max_chunks_per_doc:
                continue
            limited_scored.append((idx, score))
            per_doc_counts[doc_id] += 1
            if len(limited_scored) >= top_k:
                break
        scored = limited_scored
    else:
        scored = scored[:top_k]

    results = []
    seen_snippets: List[List[str]] = []

    for idx, base_score in scored:
        item = items[idx]

        item_latent = None
        if retrieval_mode == "svd" and svd_matrix is not None:
            item_latent = svd_matrix[idx]

        result = build_result_object(
            query=query,
            item=item,
            docs=docs,
            base_score=base_score,
            retrieval_mode=retrieval_mode,
            resolved_comedian=resolved_comedian,
            word_to_index=word_to_index,
            idf=idf,
            is_full_transcript=(result_scope == "full"),
            q_latent=q_latent,
            item_latent=item_latent,
            dimension_terms=dimension_terms,
        )

        if result_scope == "chunks":
            snippet_sentences = result.get("snippet_sentences", [])
            if any(snippets_overlap_enough(snippet_sentences, prev) for prev in seen_snippets):
                continue
            seen_snippets.append(snippet_sentences)

        results.append(result)

        if len(results) >= top_k:
            break

    return {
        "query": query,
        "results": results,
        "resolved_comedian": resolved_comedian,
        "known_comedians": known_comedians,
        "known_special_types": sorted({item.get("special_type", "") for item in items if item.get("special_type", "")}),
    }