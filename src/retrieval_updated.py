import math
import os
import re
import time
# import gzip  # teammate's compression approach (see commented save/load below)
from collections import Counter, defaultdict
from difflib import get_close_matches
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from index_builder import (
    build_transcript_index_payload,
    build_streaming_lookup_keys,
    choose_watch_link,
    clean_and_tokenize_text,
    is_garbage_token,
    load_streaming_links,
    PROFANITY_WORDS,
    STREAMING_LINKS_PATH,
)
from chunk_index_builder import build_semantic_chunks

# Streaming links cached in memory — reloaded only when the file changes
_streaming_links_cache: Dict[str, Dict[str, str]] = {}
_streaming_links_mtime: float = 0.0


def get_streaming_links() -> Dict[str, Dict[str, str]]:
    global _streaming_links_cache, _streaming_links_mtime
    try:
        mtime = os.path.getmtime(STREAMING_LINKS_PATH)
    except OSError:
        return {}
    if mtime != _streaming_links_mtime:
        _streaming_links_cache = load_streaming_links()
        _streaming_links_mtime = mtime
    return _streaming_links_cache


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRANSCRIPTS_PATH = os.path.join(DATA_DIR, "4300_transcripts_2.json")

TRANSCRIPT_DOCS_PATH = os.path.join(DATA_DIR, "transcript_docs.pkl")
TRANSCRIPT_INDEX_PATH = os.path.join(DATA_DIR, "transcript_search_index.pkl")
CHUNK_INDEX_PATH = os.path.join(DATA_DIR, "chunk_search_index.pkl")

RERANK_CANDIDATES_FULL = 300
RERANK_CANDIDATES_CHUNKS = 400

MIN_DF = 1
TRANSCRIPT_MAX_DF_RATIO = 0.95
CHUNK_MAX_DF_RATIO = 0.35

DEFAULT_TOP_K = 25
DEFAULT_SVD_COMPONENTS = 130
DEFAULT_MAX_CHUNKS_PER_DOC = 2
DEFAULT_RESULT_SCOPE = "full"

SNIPPET_WINDOW = 2
MIN_SHARED_SNIPPET_SENTENCES = 2
MIN_DISPLAY_SNIPPET_SENTENCES = 5
TARGET_DISPLAY_SNIPPET_SENTENCES = 7
MIN_DISPLAY_SNIPPET_WORDS = 85

SVD_EXPLAIN_TOP_DIMS = 3
SVD_EXPLAIN_TOP_TERMS = 6

BUILD_CHUNK_INDEX_AT_STARTUP = True
DEFAULT_USE_PROXIMITY_SCORING = True
DEFAULT_SHOW_SVD_EXPLANATIONS = True

BASE_SCORE_WEIGHT = 0.70
PROXIMITY_SCORE_WEIGHT = 0.30

_SEARCH_INDEX: Dict[str, Optional[Dict[str, Any]]] = {
    "transcript": None,
    "chunk": None,
}

def get_rerank_candidate_limit(result_scope: str) -> int:
    return RERANK_CANDIDATES_FULL if result_scope == "full" else RERANK_CANDIDATES_CHUNKS



def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def pickle_exists(path: str) -> bool:
    return os.path.exists(path)


def save_pickle(path: str, obj: Any) -> None:
    ensure_data_dir()
    joblib.dump(obj, path, compress=3)


def load_pickle(path: str) -> Any:
    return joblib.load(path)


# ---  gzip/pickle approach  ---
# def pickle_exists(path: str) -> bool:
#     return os.path.exists(path + ".gz") or os.path.exists(path)
#
# def save_pickle(path: str, obj: Any) -> None:
#     ensure_data_dir()
#     import gzip, pickle
#     gz_path = path + ".gz"
#     with gzip.open(gz_path, "wb", compresslevel=3) as f:
#         pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
#     if os.path.exists(path):
#         os.remove(path)
#
# def load_pickle(path: str) -> Any:
#     import gzip, pickle
#     gz_path = path + ".gz"
#     if os.path.exists(gz_path):
#         with gzip.open(gz_path, "rb") as f:
#             return pickle.load(f)
#     with open(path, "rb") as f:
#         return pickle.load(f)
# ------------------------------------------------------------


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
        if is_garbage_token(term):
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

def trim_snippet_to_word_limit(text: str, max_words: int = 190) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."

def trim_snippet_sentences_to_word_limit(sentences: List[str], max_words: int = 190) -> List[str]:
    count = 0
    result = []
    for s in sentences:
        w = len(s.split())
        if count + w > max_words and result:
            break
        result.append(s)
        count += w
    return result or sentences[:1]

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
    if np.linalg.norm(query_vec) == 0:
        return None, "", 0.0

    best_idx = None
    best_score = -1.0
    best_sentence = ""

    for i, (sentence, tokens) in enumerate(zip(sentences, sentence_tokens)):
        sent_vec = vectorize_tokens(tokens, word_to_index, idf, normalize_tf=True)
        denom = np.linalg.norm(query_vec) * np.linalg.norm(sent_vec)
        score = 0.0 if denom == 0 else float(np.dot(query_vec, sent_vec) / denom)

        if score > best_score:
            best_score = score
            best_idx = i
            best_sentence = sentence

    return best_idx, best_sentence, max(0.0, best_score)


def build_display_snippet_from_best_sentence(
    source_sentences: List[str],
    best_global_idx: Optional[int],
) -> Tuple[str, int, int, List[str]]:
    if not source_sentences:
        return "", 0, 0, []

    if best_global_idx is None:
        snippet_sentences = source_sentences[:min(len(source_sentences), TARGET_DISPLAY_SNIPPET_SENTENCES)]
        return " ".join(snippet_sentences), 0, len(snippet_sentences) - 1, snippet_sentences

    n = len(source_sentences)
    start = max(0, best_global_idx - SNIPPET_WINDOW)
    end = min(n - 1, best_global_idx + SNIPPET_WINDOW)

    snippet_sentences = source_sentences[start:end + 1]

    while (
        len(snippet_sentences) < MIN_DISPLAY_SNIPPET_SENTENCES
        or snippet_word_count(snippet_sentences) < MIN_DISPLAY_SNIPPET_WORDS
    ) and (start > 0 or end < n - 1):
        can_expand_left = start > 0
        can_expand_right = end < n - 1

        if can_expand_left:
            start -= 1
        if can_expand_right and (
            len(snippet_sentences) < TARGET_DISPLAY_SNIPPET_SENTENCES
            or snippet_word_count(snippet_sentences) < MIN_DISPLAY_SNIPPET_WORDS
        ):
            end += 1

        snippet_sentences = source_sentences[start:end + 1]

    return " ".join(snippet_sentences), start, end, snippet_sentences

def normalize_scores_to_unit_interval(scores: List[float]) -> List[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score - min_score <= 1e-12:
        if max_score <= 0:
            return [0.0 for _ in scores]
        return [1.0 for _ in scores]

    return [
        float((score - min_score) / (max_score - min_score))
        for score in scores
    ]


def compute_token_window_proximity_score(
    query_tokens: List[str],
    doc_tokens: List[str],
) -> float:
    """
    Partial proximity score in [0, 1].

    Score = (n_found / total_terms) * (n_found / min_window_size)
    Full credit when all terms appear adjacent; partial credit when only some
    terms appear or they are spread apart. Never zeroed out solely because a
    term is missing from the document.
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    distinct_query_terms = list(dict.fromkeys(query_tokens))
    total_terms = len(distinct_query_terms)
    if total_terms < 2:
        return 0.0

    positions_by_term: Dict[str, List[int]] = {t: [] for t in distinct_query_terms}
    for idx, token in enumerate(doc_tokens):
        if token in positions_by_term:
            positions_by_term[token].append(idx)

    found_by_term = {t: p for t, p in positions_by_term.items() if p}
    n_found = len(found_by_term)
    if n_found == 0:
        return 0.0

    coverage = n_found / total_terms
    if n_found == 1:
        return coverage

    merged_positions: List[Tuple[int, str]] = sorted(
        (pos, term) for term, positions in found_by_term.items() for pos in positions
    )

    needed = n_found
    have_counts: Dict[str, int] = defaultdict(int)
    have_distinct = 0
    left = 0
    min_window_size: Optional[int] = None

    for right, (right_pos, right_term) in enumerate(merged_positions):
        if have_counts[right_term] == 0:
            have_distinct += 1
        have_counts[right_term] += 1

        while have_distinct == needed and left <= right:
            left_pos, left_term = merged_positions[left]
            window_size = right_pos - left_pos + 1
            if min_window_size is None or window_size < min_window_size:
                min_window_size = window_size
            have_counts[left_term] -= 1
            if have_counts[left_term] == 0:
                have_distinct -= 1
            left += 1

    if min_window_size is None:
        return coverage

    return max(0.0, min(1.0, coverage * (n_found / float(min_window_size))))


def find_best_matching_sentence_by_proximity(
    query: str,
    sentences: List[str],
    sentence_tokens: List[List[str]],
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
) -> Tuple[Optional[int], str, float, float]:
    if not sentences or not sentence_tokens:
        return None, "", 0.0, 0.0

    query_tokens = clean_and_tokenize_text(query)
    if not query_tokens:
        return None, "", 0.0, 0.0

    best_idx = None
    best_sentence = ""
    best_score = 0.0

    for i, (sentence, tokens) in enumerate(zip(sentences, sentence_tokens)):
        score = compute_token_window_proximity_score(query_tokens, tokens)
        if score > best_score:
            best_score = score
            best_idx = i
            best_sentence = sentence

    if best_idx is None:
        return None, "", 0.0, 0.0

    return best_idx, best_sentence, 0.0, best_score

def build_display_snippet_for_chunk(
    query: str,
    item: Dict[str, Any],
    docs: List[Dict[str, Any]],
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
) -> Tuple[str, Optional[int], str, float, float, int, int, List[str], int, int]:
    doc = docs[item["doc_id"]]
    source_sentences = doc.get("sentences", [])

    chunk_sentences = item.get("chunk_sentences", [])
    sentence_start = item.get("sentence_start", 0)
    sentence_end = item.get("sentence_end", max(0, len(chunk_sentences) - 1))

    chunk_sentence_tokens = item.get("chunk_sentence_tokens", [])

    best_local_idx, best_sentence, sentence_score, proximity_score = find_best_matching_sentence_by_proximity(
        query=query,
        sentences=chunk_sentences,
        sentence_tokens=chunk_sentence_tokens,
        word_to_index=word_to_index,
        idf=idf,
    )

    if best_local_idx is None:
        snippet_sentences = chunk_sentences[:]
        return (
            " ".join(snippet_sentences),
            None,
            "",
            0.0,
            0.0,
            0,
            max(0, len(snippet_sentences) - 1),
            snippet_sentences,
            sentence_start,
            sentence_end,
        )

    best_global_idx = sentence_start + best_local_idx

    if len(chunk_sentences) <= 5:
        start = max(0, best_global_idx - 2)
        end = min(len(source_sentences) - 1, best_global_idx + 2)
        snippet_sentences = source_sentences[start:end + 1]
    else:
        target_len = max(MIN_DISPLAY_SNIPPET_SENTENCES, int(math.ceil(0.75 * len(chunk_sentences))))
        target_len = min(target_len, len(chunk_sentences))
        start = max(sentence_start, best_global_idx - target_len // 2)
        end = min(sentence_end, start + target_len - 1)
        start = max(sentence_start, end - target_len + 1)
        snippet_sentences = source_sentences[start:end + 1]

        while (
            (len(snippet_sentences) < MIN_DISPLAY_SNIPPET_SENTENCES or snippet_word_count(snippet_sentences) < MIN_DISPLAY_SNIPPET_WORDS)
            and (start > 0 or end < len(source_sentences) - 1)
        ):
            if start > 0:
                start -= 1
            if end < len(source_sentences) - 1:
                end += 1
            snippet_sentences = source_sentences[start:end + 1]

    local_start = max(0, start - sentence_start)
    local_end = max(0, end - sentence_start)

    return (
        " ".join(snippet_sentences),
        best_global_idx,
        best_sentence,
        sentence_score,
        proximity_score,
        local_start,
        local_end,
        snippet_sentences,
        start,
        end,
    )


def snippets_overlap_enough(a: List[str], b: List[str], min_shared: int = MIN_SHARED_SNIPPET_SENTENCES) -> bool:
    if not a or not b:
        return False
    return len(set(a) & set(b)) >= min_shared


def get_known_comedians(items: List[Dict[str, Any]]) -> List[str]:
    return sorted({item.get("comedian", "") for item in items if item.get("comedian", "")})


def resolve_comedian_name(comedian: Optional[str], known_comedians: List[str]) -> Optional[str]:
    if not comedian:
        return None

    comedian = comedian.strip()
    if not comedian:
        return None

    exact = [c for c in known_comedians if c.lower() == comedian.lower()]
    if exact:
        return exact[0]

    matches = get_close_matches(comedian, known_comedians, n=1, cutoff=0.75)
    return matches[0] if matches else comedian


def extract_comedian_from_query(query: str, known_comedians: List[str]) -> Tuple[Optional[str], str]:
    for comedian in sorted(known_comedians, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(comedian) + r'\b', re.IGNORECASE)
        if pattern.search(query):
            stripped = pattern.sub('', query).strip()
            return comedian, stripped
    return None, query


def item_passes_filters(
    item: Dict[str, Any],
    resolved_comedian: Optional[str],
    year_min: Optional[int],
    year_max: Optional[int],
    special_type: Optional[str],
    exclude_profanity: bool,
) -> bool:
    if resolved_comedian and item.get("comedian", "").lower() != resolved_comedian.lower():
        return False

    if special_type and item.get("special_type", "") != special_type:
        return False

    if exclude_profanity and item.get("has_profanity", False):
        return False

    if year_min is not None or year_max is not None:
        release_date = str(item.get("release_date", "")).strip()
        if not release_date.isdigit():
            # Unknown year — exclude when a year filter is active
            return False
        year = int(release_date)
        if year_min is not None and year < year_min:
            return False
        if year_max is not None and year > year_max:
            return False

    return True


def build_svd_dimension_terms(
    svd_model: TruncatedSVD,
    index_to_word: Dict[int, str],
    top_terms: int = SVD_EXPLAIN_TOP_TERMS,
) -> Dict[int, Dict[str, List[str]]]:
    dimension_terms: Dict[int, Dict[str, List[str]]] = {}
    candidates = top_terms * 4

    for dim_idx, component in enumerate(svd_model.components_):
        sorted_indices = np.argsort(component)
        top_negative = [index_to_word[i] for i in sorted_indices[:candidates]]
        top_positive = [index_to_word[i] for i in sorted_indices[-candidates:][::-1]]

        dimension_terms[dim_idx] = {
            "positive": top_positive,
            "negative": top_negative,
        }

    return dimension_terms


def explain_svd_alignment(
    q_latent: Optional[np.ndarray],
    item_latent: Optional[np.ndarray],
    dimension_terms: Optional[Dict[int, Dict[str, List[str]]]],
    top_dims: int = SVD_EXPLAIN_TOP_DIMS,
    exclude_profanity: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if q_latent is None or item_latent is None or dimension_terms is None:
        return [], []

    def _filter_terms(terms: List[str]) -> List[str]:
        if not exclude_profanity:
            return terms[:SVD_EXPLAIN_TOP_TERMS]
        return [t for t in terms if t.lower() not in PROFANITY_WORDS][:SVD_EXPLAIN_TOP_TERMS]

    contributions = q_latent * item_latent
    ranked = np.argsort(np.abs(contributions))[::-1]

    positive_dims = []
    negative_dims = []

    for dim in ranked:
        contribution = float(contributions[dim])
        if contribution == 0:
            continue

        dim_data = dimension_terms.get(int(dim), {})
        entry = {
            "dimension": int(dim),
            "query_weight": float(q_latent[dim]),
            "chunk_weight": float(item_latent[dim]),
            "contribution": contribution,
            "direction": "positive" if contribution > 0 else "negative",
            "top_positive_terms": _filter_terms(dim_data.get("positive", [])),
            "top_negative_terms": _filter_terms(dim_data.get("negative", [])),
        }

        if contribution > 0 and len(positive_dims) < top_dims:
            positive_dims.append(entry)
        elif contribution < 0 and len(negative_dims) < top_dims:
            negative_dims.append(entry)

        if len(positive_dims) >= top_dims and len(negative_dims) >= top_dims:
            break

    return positive_dims, negative_dims


def load_or_build_transcript_docs() -> Dict[str, Any]:
    start = time.perf_counter()

    if pickle_exists(TRANSCRIPT_DOCS_PATH):
        #print("Loading transcript docs...")
        docs_payload = load_pickle(TRANSCRIPT_DOCS_PATH)
        return docs_payload

    docs_payload = build_transcript_index_payload(TRANSCRIPTS_PATH)

    save_start = time.perf_counter()
    save_pickle(TRANSCRIPT_DOCS_PATH, docs_payload)

    return docs_payload


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
            "yt": doc.get("yt", ""),
            "raw_tags": doc.get("raw_tags", []),
            "tag_texts": doc.get("tag_texts", []),
            "platform": doc.get("platform", ""),
            "watch_url": doc.get("watch_url", ""),
            "watch_platform": doc.get("watch_platform", ""),
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
    start = time.perf_counter()

    docs_payload = load_or_build_transcript_docs()

    build_start = time.perf_counter()
    index = build_transcript_search_index_from_docs(docs_payload)

    save_start = time.perf_counter()
    save_pickle(TRANSCRIPT_INDEX_PATH, index)

    return index


def load_or_build_transcript_search_index() -> Dict[str, Any]:
    if pickle_exists(TRANSCRIPT_INDEX_PATH):
        #print("Loading transcript search index...")
        return load_pickle(TRANSCRIPT_INDEX_PATH)
    return build_transcript_search_index()


def build_chunk_search_index() -> Dict[str, Any]:
    start = time.perf_counter()


    docs_load_start = time.perf_counter()
    docs_payload = load_or_build_transcript_docs()
    docs = docs_payload["docs"]

    sentence_count = sum(len(doc.get("sentences", [])) for doc in docs)
    token_count = sum(len(doc.get("tokens", [])) for doc in docs)

    chunk_start = time.perf_counter()
    chunks, adjacent_similarities, transcript_chunk_ids = build_semantic_chunks(docs)

    if adjacent_similarities:
        avg_adj = sum(adjacent_similarities) / len(adjacent_similarities)

    tfidf_start = time.perf_counter()

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

    svd_start = time.perf_counter()
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

    save_start = time.perf_counter()
    save_pickle(CHUNK_INDEX_PATH, index)

    return index


def load_or_build_chunk_search_index() -> Dict[str, Any]:
    if pickle_exists(CHUNK_INDEX_PATH):
        #print("Loading chunk search index...")
        return load_pickle(CHUNK_INDEX_PATH)
    return build_chunk_search_index()


def initialize_search() -> None:
    global _SEARCH_INDEX

    total_start = time.perf_counter()

    transcript_start = time.perf_counter()
    _SEARCH_INDEX["transcript"] = load_or_build_transcript_search_index()

    if BUILD_CHUNK_INDEX_AT_STARTUP:
        chunk_start = time.perf_counter()
        _SEARCH_INDEX["chunk"] = load_or_build_chunk_search_index()


def reinitialize_search() -> None:
    global _SEARCH_INDEX
    _SEARCH_INDEX["transcript"] = build_transcript_search_index()
    _SEARCH_INDEX["chunk"] = build_chunk_search_index()

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


def _resolve_watch(item: Dict[str, Any]) -> Tuple[str, str]:
    url, platform = choose_watch_link(
        item.get("yt", ""),
        item.get("comedian", ""),
        item.get("special_title", ""),
        get_streaming_links(),
    )
    return url, platform


def build_result_object(
    *,
    query: str,
    item: Dict[str, Any],
    docs: List[Dict[str, Any]],
    base_score: float,
    final_score: float,
    proximity_feature: float,
    retrieval_mode: str,
    resolved_comedian: Optional[str],
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
    is_full_transcript: bool,
    include_svd_explanations: bool = True,
    exclude_profanity: bool = False,
    q_latent: Optional[np.ndarray] = None,
    item_latent: Optional[np.ndarray] = None,
    dimension_terms: Optional[Dict[int, Dict[str, List[str]]]] = None,
) -> Dict[str, Any]:
    if not query:
        if is_full_transcript:
            source_sentences = docs[item["doc_id"]].get("sentences", [])
            display_snippet, snippet_start, snippet_end, snippet_sentences = build_display_snippet_from_best_sentence(
                source_sentences=source_sentences,
                best_global_idx=None,
            )
            global_snippet_start = snippet_start
            global_snippet_end = snippet_end
        else:
            chunk_sentences = item.get("chunk_sentences", [])
            snippet_sentences = chunk_sentences[:]
            snippet_start = item.get("sentence_start", 0)
            snippet_end = item.get("sentence_end", max(0, len(chunk_sentences) - 1))
            global_snippet_start = snippet_start
            global_snippet_end = snippet_end
            display_snippet = " ".join(chunk_sentences)
        display_snippet = trim_snippet_to_word_limit(display_snippet, 190)
        best_sentence_index = None
        best_sentence = ""
        sentence_score = 0.0
        best_sentence_proximity = 0.0
    elif is_full_transcript:
        doc = docs[item["doc_id"]]
        source_sentences = doc.get("sentences", [])
        source_sentence_tokens = doc.get("sentence_tokens", [])

        best_idx, best_sentence, sentence_score, best_sentence_proximity = find_best_matching_sentence_by_proximity(
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
        display_snippet = trim_snippet_to_word_limit(display_snippet, 190)

        best_sentence_index = best_idx
        global_snippet_start = snippet_start
        global_snippet_end = snippet_end
    else:
        (
            display_snippet,
            best_sentence_index,
            best_sentence,
            sentence_score,
            _,
            snippet_start,
            snippet_end,
            snippet_sentences,
            global_snippet_start,
            global_snippet_end,
        ) = build_display_snippet_for_chunk(
            query=query,
            item=item,
            docs=docs,
            word_to_index=word_to_index,
            idf=idf,
        )

        display_snippet = trim_snippet_to_word_limit(display_snippet, 190)

    comedian_feature = 1.0 if (
        resolved_comedian
        and item.get("comedian", "").strip().lower() == resolved_comedian.strip().lower()
    ) else 0.0

    similarity_score = float(max(0.0, min(1.0, final_score)))
    similarity_percent = 100.0 * similarity_score

    _watch_url, _watch_platform = _resolve_watch(item)

    svd_positive_dimensions = None
    svd_negative_dimensions = None

    if (
        include_svd_explanations
        and retrieval_mode == "svd"
        and q_latent is not None
        and item_latent is not None
        and dimension_terms is not None
    ):
        svd_positive_dimensions, svd_negative_dimensions = explain_svd_alignment(
            q_latent=q_latent,
            item_latent=item_latent,
            dimension_terms=dimension_terms,
            top_dims=SVD_EXPLAIN_TOP_DIMS,
            exclude_profanity=exclude_profanity,
        ) if retrieval_mode == "svd" else ([], [])

    return {
        "chunk_id": item.get("chunk_id", f"doc-{item['doc_id']}"),
        "doc_id": item["doc_id"],
        "comedian": item.get("comedian", ""),
        "special_title": item.get("special_title", ""),
        "release_date": item.get("release_date", ""),
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "platform": item.get("platform", ""),
        "special_type": item.get("special_type", ""),
        "content": item.get("content", ""),
        "display_snippet": display_snippet,
        "chunk_sentences": item.get("chunk_sentences"),
        "best_sentence": best_sentence,
        "best_sentence_index": best_sentence_index,
        "sentence_score": sentence_score,
        "snippet_sentences": trim_snippet_sentences_to_word_limit(snippet_sentences, 190),
        "snippet_sentence_start": snippet_start,
        "snippet_sentence_end": snippet_end,
        "global_snippet_start": global_snippet_start,
        "global_snippet_end": global_snippet_end,
        "has_profanity": item.get("has_profanity", False),
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
        "watch_url": _watch_url,
        "watch_platform": _watch_platform,
    }

def compute_full_window_proximity_score(
    query_tokens: List[str],
    doc_tokens: List[str],
) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    distinct_query_terms = list(dict.fromkeys(query_tokens))
    total_terms = len(distinct_query_terms)
    if total_terms < 2:
        return 0.0

    positions_by_term: Dict[str, List[int]] = {t: [] for t in distinct_query_terms}
    for idx, token in enumerate(doc_tokens):
        if token in positions_by_term:
            positions_by_term[token].append(idx)

    found_by_term = {t: p for t, p in positions_by_term.items() if p}
    n_found = len(found_by_term)
    if n_found == 0:
        return 0.0

    coverage = n_found / total_terms
    if n_found == 1:
        return coverage

    merged_positions: List[Tuple[int, str]] = sorted(
        (pos, term) for term, positions in found_by_term.items() for pos in positions
    )

    needed = n_found
    have_counts: Dict[str, int] = defaultdict(int)
    have_distinct = 0
    left = 0
    min_window_size: Optional[int] = None

    for right, (right_pos, right_term) in enumerate(merged_positions):
        if have_counts[right_term] == 0:
            have_distinct += 1
        have_counts[right_term] += 1

        while have_distinct == needed and left <= right:
            left_pos, left_term = merged_positions[left]
            window_size = right_pos - left_pos + 1
            if min_window_size is None or window_size < min_window_size:
                min_window_size = window_size
            have_counts[left_term] -= 1
            if have_counts[left_term] == 0:
                have_distinct -= 1
            left += 1

    if min_window_size is None:
        return coverage

    return max(0.0, min(1.0, coverage * (n_found / float(min_window_size))))

def compute_fast_proximity_score(
    query_tokens: List[str],
    doc_tokens: List[str],
) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    distinct_query_terms = list(dict.fromkeys(query_tokens))
    total_terms = len(distinct_query_terms)
    if total_terms < 2:
        return 0.0

    first_positions = []
    found_terms = set()

    for idx, token in enumerate(doc_tokens):
        if token in distinct_query_terms and token not in found_terms:
            first_positions.append(idx)
            found_terms.add(token)

    n_found = len(found_terms)
    if n_found == 0:
        return 0.0

    coverage = n_found / total_terms
    if n_found == 1:
        return coverage

    span = max(first_positions) - min(first_positions) + 1
    return max(0.0, min(1.0, coverage * (n_found / float(span))))

def sort_indices_for_filter_only_browse(
    indices: List[int],
    items: List[Dict[str, Any]],
) -> List[int]:
    def sort_key(i: int):
        item = items[i]

        release_date_raw = str(item.get("release_date", "")).strip()
        try:
            release_year = int(release_date_raw)
        except Exception:
            release_year = -1

        comedian = str(item.get("comedian", "")).lower()
        special_title = str(item.get("special_title", "")).lower()
        title = str(item.get("title", "")).lower()

        return (
            release_year,
            comedian,
            special_title or title,
        )

    return sorted(indices, key=sort_key, reverse=True)
  
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
    use_expensive_proximity_scoring: bool = False,
    show_svd_explanations: bool = DEFAULT_SHOW_SVD_EXPLANATIONS,
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
    if not comedian:
        comedian, query = extract_comedian_from_query(query, known_comedians)
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
            "known_special_types": sorted(
                {item.get("special_type", "") for item in items if item.get("special_type", "")}
            ),
        }

    query = query.strip()

    if not query:
        ranked_indices = sort_indices_for_filter_only_browse(filtered_indices, items)

        if result_scope == "full":
            selected = ranked_indices[:top_k]
            results = []

            for idx in selected:
                item = items[idx]
                result = build_result_object(
                    query="",
                    item=item,
                    docs=docs,
                    base_score=1.0,
                    final_score=1.0,
                    proximity_feature=0.0,
                    retrieval_mode=retrieval_mode,
                    resolved_comedian=resolved_comedian,
                    word_to_index=word_to_index,
                    idf=idf,
                    is_full_transcript=True,
                    include_svd_explanations=False,
                    exclude_profanity=exclude_profanity,
                    q_latent=None,
                    item_latent=None,
                    dimension_terms=dimension_terms,
                )
                result["similarity_score"] = None
                result["similarity_percent"] = None
                result["retrieval_mode"] = retrieval_mode
                results.append(result)

            return {
                "query": query,
                "results": results,
                "resolved_comedian": resolved_comedian,
                "known_comedians": known_comedians,
                "known_special_types": sorted(
                    {item.get("special_type", "") for item in items if item.get("special_type", "")}
                ),
            }

        selected_results = []
        per_doc_counts: Dict[int, int] = defaultdict(int)
        accepted_snippets_by_doc: Dict[int, List[List[str]]] = defaultdict(list)

        for idx in ranked_indices:
            item = items[idx]
            doc_id = item["doc_id"]

            if per_doc_counts[doc_id] >= max_chunks_per_doc:
                continue

            result = build_result_object(
                query="",
                item=item,
                docs=docs,
                base_score=1.0,
                final_score=1.0,
                proximity_feature=0.0,
                retrieval_mode=retrieval_mode,
                resolved_comedian=resolved_comedian,
                word_to_index=word_to_index,
                idf=idf,
                is_full_transcript=False,
                include_svd_explanations=False,
                exclude_profanity=exclude_profanity,
                q_latent=None,
                item_latent=None,
                dimension_terms=dimension_terms,
            )
            result["similarity_score"] = None
            result["similarity_percent"] = None
            result["retrieval_mode"] = retrieval_mode

            current_snippet = result.get("snippet_sentences", []) or []
            existing_snippets = accepted_snippets_by_doc[doc_id]

            if any(snippets_overlap_enough(current_snippet, prev) for prev in existing_snippets):
                continue

            accepted_snippets_by_doc[doc_id].append(current_snippet)
            per_doc_counts[doc_id] += 1
            selected_results.append(result)

            if len(selected_results) >= top_k:
                break

        return {
            "query": query,
            "results": selected_results,
            "resolved_comedian": resolved_comedian,
            "known_comedians": known_comedians,
            "known_special_types": sorted(
                {item.get("special_type", "") for item in items if item.get("special_type", "")}
            ),
        }

    query_vec = vectorize_query(query, word_to_index, idf)
    query_tokens = clean_and_tokenize_text(query)

    if retrieval_mode == "svd" and svd_model is not None and svd_matrix is not None:
        q_latent = normalize(svd_model.transform(query_vec.reshape(1, -1)))[0]
        raw_base_scores = cosine_scores_dense(q_latent, svd_matrix)
    else:
        q_latent = None
        raw_base_scores = cosine_scores_sparse(query_vec, tfidf_matrix)

    filtered_base_scores = [float(max(0.0, raw_base_scores[i])) for i in filtered_indices]
    normalized_base_scores = normalize_scores_to_unit_interval(filtered_base_scores)
    normalized_base_score_by_idx = {
        idx: score for idx, score in zip(filtered_indices, normalized_base_scores)
    }

    base_ranked_indices = sorted(
        filtered_indices,
        key=lambda i: normalized_base_score_by_idx[i],
        reverse=True,
    )

    rerank_limit = get_rerank_candidate_limit(result_scope)
    rerank_indices = base_ranked_indices[: min(len(base_ranked_indices), rerank_limit)]

    proximity_score_by_idx: Dict[int, float] = {}
    distinct_query_terms = list(dict.fromkeys(query_tokens))

    for idx in rerank_indices:
        item = items[idx]
        item_tokens = item.get("tokens", [])

        if len(distinct_query_terms) < 2:
            proximity_score_by_idx[idx] = 0.0
        elif use_expensive_proximity_scoring:
            proximity_score_by_idx[idx] = compute_full_window_proximity_score(
                query_tokens,
                item_tokens,
            )
        else:
            proximity_score_by_idx[idx] = compute_token_window_proximity_score(
                query_tokens,
                item_tokens,
            )

    final_score_by_idx = {
        idx: (
            BASE_SCORE_WEIGHT * normalized_base_score_by_idx[idx]
            + PROXIMITY_SCORE_WEIGHT * proximity_score_by_idx[idx]
        )
        for idx in rerank_indices
    }

    ranked_indices = sorted(
        rerank_indices,
        key=lambda i: (
            final_score_by_idx[i],
            proximity_score_by_idx[i],
            normalized_base_score_by_idx[i],
        ),
        reverse=True,
    )


    if result_scope == "full":
        selected = ranked_indices[:top_k]
        results = []

        for idx in selected:
            item = items[idx]
            item_latent = svd_matrix[idx] if retrieval_mode == "svd" and svd_matrix is not None else None

            result = build_result_object(
                query=query,
                item=item,
                docs=docs,
                base_score=float(normalized_base_score_by_idx[idx]),
                final_score=float(final_score_by_idx[idx]),
                proximity_feature=float(proximity_score_by_idx[idx]),
                retrieval_mode=retrieval_mode,
                resolved_comedian=resolved_comedian,
                word_to_index=word_to_index,
                idf=idf,
                is_full_transcript=True,
                include_svd_explanations=show_svd_explanations,
                exclude_profanity=exclude_profanity,
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
            "known_special_types": sorted(
                {item.get("special_type", "") for item in items if item.get("special_type", "")}
            ),
        }

    selected_results = []
    per_doc_counts: Dict[int, int] = defaultdict(int)
    accepted_snippets_by_doc: Dict[int, List[List[str]]] = defaultdict(list)

    for idx in ranked_indices:
        item = items[idx]
        doc_id = item["doc_id"]

        if per_doc_counts[doc_id] >= max_chunks_per_doc:
            continue

        item_latent = svd_matrix[idx] if retrieval_mode == "svd" and svd_matrix is not None else None

        result = build_result_object(
            query=query,
            item=item,
            docs=docs,
            base_score=float(normalized_base_score_by_idx[idx]),
            final_score=float(final_score_by_idx[idx]),
            proximity_feature=float(proximity_score_by_idx[idx]),
            retrieval_mode=retrieval_mode,
            resolved_comedian=resolved_comedian,
            word_to_index=word_to_index,
            idf=idf,
            is_full_transcript=False,
            include_svd_explanations=show_svd_explanations,
            exclude_profanity=exclude_profanity,
            q_latent=q_latent,
            item_latent=item_latent,
            dimension_terms=dimension_terms,
        )

        existing_snippets = accepted_snippets_by_doc[doc_id]
        current_snippet = result.get("snippet_sentences", []) or []

        if any(snippets_overlap_enough(current_snippet, prev) for prev in existing_snippets):
            continue

        selected_results.append(result)
        accepted_snippets_by_doc[doc_id].append(current_snippet)
        per_doc_counts[doc_id] += 1

        if len(selected_results) >= top_k:
            break

    return {
        "query": query,
        "results": selected_results,
        "resolved_comedian": resolved_comedian,
        "known_comedians": known_comedians,
        "known_special_types": sorted(
            {item.get("special_type", "") for item in items if item.get("special_type", "")}
        ),
    }
