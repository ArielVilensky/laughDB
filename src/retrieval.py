import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "search_index.pkl")
TRANSCRIPTS_PATH = os.path.join(DATA_DIR, "4300_transcripts.json")


# -------------------------------------------------------------------
# Text cleaning / preprocessing helpers
# -------------------------------------------------------------------

def parse_title_metadata(title: str) -> Tuple[str, str, str]:

    comedian = ""
    special_title = ""
    release_date = ""

    cleaned_title = re.sub(
        r"\s*[|\-\–:]?\s*[(\[{]?(?:Full )?Transcript[)\]}]?\s*[|\-\–:]?\s*",
        " ",
        title,
        flags=re.IGNORECASE
    ).strip()

    year_match = re.search(r"(?:20|19)\d\d|(?:\d{1,2}\/){1,2}\d{2,4}", cleaned_title)

    if year_match:
        release_date = year_match.group()
        cleaned_title = re.sub(rf"\s*[|\-\–:]?\s*(?:[(\[{{].*)?{release_date}(?:.*[)\]}}])?\s*[|\-\–:]?\s*", " ", cleaned_title).strip()

    if ":" in cleaned_title:
        parts = cleaned_title.split(":", 1)
        comedian = parts[0].strip()
        special_title = parts[1].strip()
    elif " on " in cleaned_title.lower():
        parts = re.split(r"(?i)\s+on\s+", cleaned_title, maxsplit=1)
        comedian = parts[0].strip()
        special_title = cleaned_title
    elif " at " in cleaned_title.lower():
        parts = re.split(r"(?i)\s+at\s+", cleaned_title, maxsplit=1)
        comedian = parts[0].strip()
        special_title = cleaned_title
    else:
        special_title = cleaned_title

    return comedian, special_title, release_date





def infer_platform(url: str, title: str = "") -> str:
    text = f"{url} {title}".lower()

    if "netflix" in text:
        return "Netflix"
    if "hbo" in text:
        return "HBO"
    if "comedy central" in text:
        return "Comedy Central"
    if "amazon" in text or "prime" in text:
        return "Amazon Prime"
    if "youtube" in text:
        return "YouTube"

    return ""


def restructure_transcripts(transcripts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    structured = []

    for t in transcripts:
        title = t.get("title", "")
        url = t.get("url", "")
        content = t.get("content", "")

        comedian, special_title, release_date = parse_title_metadata(title)

        structured.append({
            "url": url,
            "title": title,
            "comedian": comedian,
            "special_title": special_title,
            "release_date": release_date,
            "director": "",
            "platform": infer_platform(url, title),
            "content": content,
            "tokens": [],
            "length": 0,
        })

    return structured


def bracket_if_valid(brackets: str) -> str:
    inside_str = brackets[1:-1].lower().replace('\n', '')
    if re.match(r"^((audience|crowd)( member(s)?)?|all|laughs)$", inside_str):
        return brackets
    elif re.match(r".*(applau|cheer|laugh|audience|crowd|all|clap|whoo[^sh]|music).*", inside_str):
        return ' '
    return brackets


def remove_bracketed_descriptions(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", lambda match : bracket_if_valid(match.group()), text)


def normalize_text(text: str) -> str:
    text = remove_bracketed_descriptions(text)
    text = re.sub(r"‘|’", "'", text)
    text = re.sub(r"“|”", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"(?:\d+/)*\d+|[a-zA-Z0-9]+", text)


def merge_spaced_letters(text):
    """
    Example: "m i d g e t" --> "midget"
    """
    return re.sub(
        r'\b(?:[A-Za-z]\s+){1,}[A-Za-z]\b',
        lambda match: re.sub(r'\s+', '', match.group(0)),
        text
    )


def remove_stop_words(tokens: List[str]) -> List[str]:
    return [tok for tok in tokens if tok not in ENGLISH_STOP_WORDS]


def clean_and_tokenize_text(text: str) -> List[str]:
    text = merge_spaced_letters(normalize_text(text))
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    return tokens


def add_clean_tokens_to_transcripts(transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for doc in transcripts:
        toks = clean_and_tokenize_text(doc["content"])
        doc["tokens"] = toks
        doc["length"] = len(toks)
    return transcripts


def build_word_document_count(transcripts: List[Dict[str, Any]]) -> Dict[str, int]:
    doc_count = Counter()

    for doc in transcripts:
        unique_tokens = set(doc["tokens"])
        doc_count.update(unique_tokens)

    return dict(doc_count)


def build_good_words(
    transcripts: List[Dict[str, Any]],
    min_df: int = 2,
    max_df_ratio: float = 0.80
) -> List[str]:
    doc_count = build_word_document_count(transcripts)
    n_docs = len(transcripts)

    good_words = [
        word for word, df in doc_count.items()
        if df >= min_df and (df / n_docs) <= max_df_ratio
    ]

    return sorted(good_words)


def filter_tokens_to_good_words(
    transcripts: List[Dict[str, Any]],
    good_words: List[str]
) -> List[Dict[str, Any]]:
    good_set = set(good_words)

    for doc in transcripts:
        filtered = [tok for tok in doc["tokens"] if tok in good_set]
        doc["tokens"] = filtered
        doc["length"] = len(filtered)

    return transcripts


def build_inverted_index(transcripts: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, int]]]:
    index = defaultdict(list)

    for doc_id, doc in enumerate(transcripts):
        counts = Counter(doc["tokens"])
        for term, tf in counts.items():
            index[term].append((doc_id, tf))

    return dict(index)


def compute_idf(
    inv_idx: Dict[str, List[Tuple[int, int]]],
    n_docs: int,
    min_df: int = 2,
    max_df_ratio: float = 0.80
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
    transcripts: List[Dict[str, Any]],
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
    normalize_tf: bool = False
) -> Any:
    n_docs = len(transcripts)
    vocab_size = len(word_to_index)
    
    rows = []
    cols = []
    data = []

    for doc_id, doc in enumerate(transcripts):
        counts = Counter(doc["tokens"])
        doc_len = len(doc["tokens"])

        for term, raw_tf in counts.items():
            if term not in word_to_index:
                continue

            if normalize_tf:
                tf = raw_tf / doc_len if doc_len else 0.0
            else:
                tf = raw_tf

            rows.append(doc_id)
            cols.append(word_to_index[term])
            data.append(tf * idf[term])

    mat = sp.csr_matrix((data, (rows, cols)), shape=(n_docs, vocab_size), dtype=float)
    return mat


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


def split_transcript_into_sentences(content: str) -> List[str]:
    text = normalize_text(content)
    sentences = re.split(r'(?<!\w\.\w.)(?<!\b[A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?)\s|\\n', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    refined = []
    for s in sentences:
        if len(s.split()) > 60:
            parts = re.split(r'(?<=[;:])\s+', s)
            refined.extend([p.strip() for p in parts if p.strip()])
        else:
            refined.append(s)

    return refined


# -------------------------------------------------------------------
# Index build / load
# -------------------------------------------------------------------

def load_raw_transcripts(path: str = TRANSCRIPTS_PATH) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    transcripts = []

    decoder = json.JSONDecoder()
    idx = 0
    n = len(raw_text)

    while idx < n:
        while idx < n and raw_text[idx].isspace():
            idx += 1

        if idx >= n:
            break

        obj, next_idx = decoder.raw_decode(raw_text, idx)

        transcripts.append({
            "url": obj.get("url", ""),
            "title": obj.get("title", ""),
            "content": obj.get("content", ""),
        })

        idx = next_idx

    return transcripts

def build_search_index(
    transcripts_path: str = TRANSCRIPTS_PATH,
    index_path: str = INDEX_PATH,
    min_df: int = 2,
    max_df_ratio: float = 0.80
) -> Dict[str, Any]:
    raw_transcripts = load_raw_transcripts(transcripts_path)
    transcripts = restructure_transcripts(raw_transcripts)
    transcripts = add_clean_tokens_to_transcripts(transcripts)

    good_words = build_good_words(transcripts, min_df=min_df, max_df_ratio=max_df_ratio)
    transcripts = filter_tokens_to_good_words(transcripts, good_words)

    inv_idx = build_inverted_index(transcripts)
    idf = compute_idf(inv_idx, len(transcripts), min_df=min_df, max_df_ratio=max_df_ratio)
    vocab, word_to_index, _ = create_vocab(idf)
    tfidf_matrix = create_tfidf_matrix(transcripts, word_to_index, idf)

    payload = {
        "transcripts": transcripts,
        "idf": idf,
        "word_to_index": word_to_index,
        "vocab": vocab,
        "tfidf_matrix": tfidf_matrix,
    }

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(payload, f)

    return payload


def load_search_index(index_path: str = INDEX_PATH) -> Dict[str, Any]:
    with open(index_path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------------------------
# Retrieval
# -------------------------------------------------------------------

def retrieve_by_cosine(
    query: str,
    tfidf_matrix: Any,
    transcripts: List[Dict[str, Any]],
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
    top_k: int = 10
) -> List[Tuple[float, int]]:
    q_vec = vectorize_query(query, word_to_index, idf)
    q_norm = np.linalg.norm(q_vec)

    if q_norm == 0:
        return []

    if sp.issparse(tfidf_matrix):
        numerators = tfidf_matrix.dot(q_vec)
        doc_norms = np.sqrt(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).A1
    else:
        doc_norms = np.linalg.norm(tfidf_matrix, axis=1)
        numerators = tfidf_matrix @ q_vec

    scores = []
    for doc_id in range(len(transcripts)):
        denom = doc_norms[doc_id] * q_norm
        if denom == 0:
            continue

        score = numerators[doc_id] / denom
        scores.append((float(score), doc_id))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


def find_best_matching_sentence_window(
    query: str,
    transcript: Dict[str, Any],
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
    match_window: int = 1
) -> Tuple[int | None, float, List[str]]:
    sentences = split_transcript_into_sentences(transcript["content"])

    if not sentences:
        return None, 0.0, []

    query_vec = vectorize_query(query, word_to_index, idf)
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        return None, 0.0, sentences

    best_idx = None
    best_score = -1.0

    for i in range(len(sentences)):
        start = max(0, i - match_window)
        end = min(len(sentences), i + match_window + 1)

        window_text = " ".join(sentences[start:end])
        window_tokens = clean_and_tokenize_text(window_text)
        window_vec = vectorize_tokens(window_tokens, word_to_index, idf)

        denom = np.linalg.norm(window_vec) * query_norm
        if denom == 0:
            continue

        score = float(np.dot(window_vec, query_vec) / denom)

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, best_score, sentences


def get_sentence_context(sentences: List[str], center_idx: int | None, window: int = 3) -> List[str]:
    if center_idx is None:
        return []

    start = max(0, center_idx - window)
    end = min(len(sentences), center_idx + window + 1)
    return sentences[start:end]


def retrieve_top_transcripts_with_sentence_context(
    query: str,
    transcripts: List[Dict[str, Any]],
    tfidf_matrix: Any,
    word_to_index: Dict[str, int],
    idf: Dict[str, float],
    top_k: int = 5,
    context_window: int = 3,
    match_window: int = 1
) -> List[Dict[str, Any]]:
    top_docs = retrieve_by_cosine(
        query=query,
        tfidf_matrix=tfidf_matrix,
        transcripts=transcripts,
        word_to_index=word_to_index,
        idf=idf,
        top_k=top_k
    )

    results = []

    for transcript_score, doc_id in top_docs:
        transcript = transcripts[doc_id]

        best_idx, sentence_score, sentences = find_best_matching_sentence_window(
            query=query,
            transcript=transcript,
            word_to_index=word_to_index,
            idf=idf,
            match_window=match_window
        )

        context_sentences = get_sentence_context(
            sentences=sentences,
            center_idx=best_idx,
            window=context_window
        )

        best_sentence = (
            sentences[best_idx]
            if best_idx is not None and 0 <= best_idx < len(sentences)
            else ""
        )

        results.append({
            "doc_id": int(doc_id),
            "transcript_score": float(transcript_score),
            "sentence_score": float(sentence_score),
            "title": transcript["title"],
            "comedian": transcript["comedian"],
            "special_title": transcript["special_title"],
            "release_date": transcript["release_date"],
            "platform": transcript.get("platform", ""),
            "url": transcript.get("url", ""),
            "best_sentence_index": None if best_idx is None else int(best_idx),
            "best_sentence": best_sentence,
            "context_sentences": context_sentences,
            "context": " ".join(context_sentences),
        })

    return results


# -------------------------------------------------------------------
# Global search engine state
# -------------------------------------------------------------------

_SEARCH_DATA: Dict[str, Any] | None = None


def initialize_search() -> None:
    global _SEARCH_DATA

    if _SEARCH_DATA is not None:
        return

    rebuild = True
    if os.path.exists(INDEX_PATH) and os.path.exists(TRANSCRIPTS_PATH):
        index_mtime = os.path.getmtime(INDEX_PATH)
        transcripts_mtime = os.path.getmtime(TRANSCRIPTS_PATH)
        if index_mtime >= transcripts_mtime:
            rebuild = False

    if not rebuild:
        try:
            _SEARCH_DATA = load_search_index(INDEX_PATH)
        except Exception:
            _SEARCH_DATA = build_search_index()
    else:
        _SEARCH_DATA = build_search_index()


def search(
    query: str,
    top_k: int = 5,
    context_window: int = 3,
    match_window: int = 1
) -> List[Dict[str, Any]]:
    initialize_search()

    assert _SEARCH_DATA is not None

    if not query or not query.strip():
        return []

    return retrieve_top_transcripts_with_sentence_context(
        query=query,
        transcripts=_SEARCH_DATA["transcripts"],
        tfidf_matrix=_SEARCH_DATA["tfidf_matrix"],
        word_to_index=_SEARCH_DATA["word_to_index"],
        idf=_SEARCH_DATA["idf"],
        top_k=top_k,
        context_window=context_window,
        match_window=match_window,
    )