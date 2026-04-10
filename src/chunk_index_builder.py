import json
import re
import statistics
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer


# -----------------------------
# CONFIG
# -----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MIN_CHUNK_SENTENCES = 3
MIN_CHUNK_WORDS = 80
SEMANTIC_BREAK_THRESHOLD = 0.10
DEFAULT_SAVE_DEBUG_JSON = False
ENABLE_DEBUG_PRINTS = False


# -----------------------------
# GLOBAL EMBEDDING MODEL CACHE
# -----------------------------
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None
_EMBEDDING_MODEL_NAME: Optional[str] = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_NAME

    if _EMBEDDING_MODEL is None or _EMBEDDING_MODEL_NAME != model_name:
        print(f"Loading embedding model: {model_name}")
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
        _EMBEDDING_MODEL_NAME = model_name

    return _EMBEDDING_MODEL


# -----------------------------
# PROFANITY WORDS
# -----------------------------
PROFANITY_WORDS = {
    "fuck", "fucking", "fucked", "fucker",
    "shit", "shitty",
    "bitch", "bitches", "bitching",
    "asshole", "motherfucker", "dick",
    "pussy", "cunt", "cock", "cocksucker",
    "bastard", "damn",
}


# -----------------------------
# CUSTOM STOPWORDS
# -----------------------------
CUSTOM_STOPWORDS = {
    "im", "ive", "ill", "id",
    "youre", "youve", "youll", "youd",
    "were", "weve", "well", "wed",
    "theyre", "theyve", "theyll", "theyd",
    "thats", "theres", "whats", "wheres", "whos", "hows",
    "dont", "didnt", "doesnt", "isnt", "arent", "wasnt", "werent",
    "wouldnt", "couldnt", "shouldnt", "cant", "wont", "aint",
    "gonna", "wanna", "gotta",
    "uh", "um", "oh", "yeah", "hey", "ok", "okay",
    "like", "know", "just", "really", "mean", "right"
}


# -----------------------------
# RAW LOADING
# -----------------------------
def load_raw_transcripts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    transcripts = []

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    jsonl_ok = True
    tmp = []
    for line in lines:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                tmp.append(obj)
            else:
                jsonl_ok = False
                break
        except Exception:
            jsonl_ok = False
            break

    if jsonl_ok and tmp:
        for obj in tmp:
            transcripts.append({
                "url": obj.get("url", ""),
                "title": obj.get("title", ""),
                "content": obj.get("content", ""),
            })
        return transcripts

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


# -----------------------------
# TEXT CLEANING
# -----------------------------
def remove_bracketed_descriptions(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", " ", text)


def normalize_decades(text: str) -> str:
    text = re.sub(r"\b(\d{2})['’]s\b", r"\1s", text)
    text = re.sub(r"\b(\d{4})['’]s\b", r"\1s", text)
    return text


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = remove_bracketed_descriptions(text)
    text = re.sub(r"♪[^♪]*♪", " ", text)
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("\xa0", " ")
    text = normalize_decades(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_structure(text: str) -> str:
    if not text:
        return ""

    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("\xa0", " ")
    text = normalize_decades(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_for_flags(text: str) -> List[str]:
    text = normalize_text(text).lower()
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)
    return [tok.replace("'", "") for tok in tokens]


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)
    return [tok.replace("'", "") for tok in tokens]


def is_noise_token(tok: str) -> bool:
    if len(tok) >= 5 and len(set(tok)) <= 2:
        return True
    if re.search(r"(.)\1{3,}", tok):
        return True
    return False


def remove_stop_words(tokens: List[str]) -> List[str]:
    stopwords = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)
    filtered = []

    for tok in tokens:
        if tok in stopwords:
            continue
        if len(tok) <= 1:
            continue
        if is_noise_token(tok):
            continue
        filtered.append(tok)

    return filtered


def clean_and_tokenize_text(text: str) -> List[str]:
    text = normalize_text(text)
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    return tokens


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


# -----------------------------
# TRANSCRIPT CLEANING
# -----------------------------
STAR_BREAK_RE = re.compile(r"\*\s*\*\s*\*")

OPENING_NOISE_LINE_PATTERNS = [
    r"^\s*\[[^\]]+\]\s*$",
    r"^\s*[A-Z][A-Z\s,&!'\-]{6,}\s*$",
    r"^\s*(thank you(?: very much)?[!. ]*){1,4}\s*$",
    r"^\s*(cheering|applause|laughter|crowd cheering|crowd cheering and applause|music)\s*$",
]

OPENING_NOISE_SENTENCE_PATTERNS = [
    r"^\s*\[[^\]]+\]\s*$",
    r"^\s*(ladies and gentlemen[,! ]*)?(please welcome\b.*)$",
    r"^\s*(and now[,! ]*)?(live from\b.*)$",
    r"^\s*(thank you(?: very much)?|hello|good evening|how are you|what's up|whats up|sit down)[!. ]*\s*$",
]


def strip_before_star_break(text: str) -> str:
    if not text:
        return ""

    match = STAR_BREAK_RE.search(text)
    if not match:
        return text.strip()

    suffix = text[match.end():].strip()
    return suffix if suffix else text.strip()


def is_opening_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True

    for pattern in OPENING_NOISE_LINE_PATTERNS:
        if re.match(pattern, stripped, flags=re.IGNORECASE):
            return True

    return False


def is_opening_noise_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return True

    for pattern in OPENING_NOISE_SENTENCE_PATTERNS:
        if re.match(pattern, stripped, flags=re.IGNORECASE):
            return True

    return False


def trim_opening_noise(text: str, max_lines_to_trim: int = 12, max_sentences_to_trim: int = 6) -> str:
    if not text:
        return ""

    working = normalize_for_structure(text)

    lines = working.split("\n")
    trimmed_lines = 0
    while lines and trimmed_lines < max_lines_to_trim and is_opening_noise_line(lines[0]):
        lines.pop(0)
        trimmed_lines += 1

    working = "\n".join(lines).strip()
    if not working:
        return ""

    sentences = split_text_into_sentences(working)
    trimmed_sentences = 0
    while sentences and trimmed_sentences < max_sentences_to_trim and is_opening_noise_sentence(sentences[0]):
        sentences.pop(0)
        trimmed_sentences += 1

    return " ".join(sentences).strip() if sentences else working.strip()


def clean_transcript_content(content: str, title: str = "") -> str:
    if not content:
        return ""

    text = normalize_for_structure(content)

    # strict rule: if *** exists and there is text after it, drop everything before it
    text = strip_before_star_break(text)

    # then trim obvious transcript-opening noise
    text = trim_opening_noise(text)

    return text.strip()


# -----------------------------
# PROFANITY DETECTION
# -----------------------------
def detect_profanity(tokens: List[str]) -> Tuple[bool, List[str]]:
    found = sorted({tok for tok in tokens if tok in PROFANITY_WORDS})
    return (len(found) > 0, found)


# -----------------------------
# METADATA INFERENCE
# -----------------------------
def parse_title_metadata(title: str) -> Tuple[str, str, str]:
    if not title:
        return "", "", ""

    t = normalize_text(title)

    t = re.sub(r"\|\s*transcript\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[-|]\s*full transcript\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[-|]\s*transcripci[oó]n completa\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[-|]\s*traduzione italiana\s*$", "", t, flags=re.IGNORECASE)

    release_date = ""
    year_match = re.search(r"\((\d{4})\)", t)
    if year_match:
        release_date = year_match.group(1)
        t = re.sub(r"\s*\(\d{4}\)", "", t).strip()

    comedian = ""
    special_title = t

    if ":" in t:
        left, right = t.split(":", 1)
        comedian = left.strip()
        special_title = right.strip()
    else:
        m = re.match(r"^([A-Z][A-Za-z\.\'\- ]+?)\s+[-–]\s+(.+)$", t)
        if m:
            comedian = m.group(1).strip()
            special_title = m.group(2).strip()

    return comedian, special_title, release_date


def infer_platform(url: str = "", title: str = "", content: str = "") -> str:
    text = normalize_text(f"{url} {title} {content[:2000]}").lower()

    if "netflix" in text:
        return "Netflix"
    if "hbo max" in text or "hbo" in text:
        return "HBO"
    if "comedy central" in text:
        return "Comedy Central"
    if "amazon prime" in text or "prime video" in text:
        return "Amazon Prime"
    if "youtube" in text:
        return "YouTube"
    if "hulu" in text:
        return "Hulu"
    if "apple tv+" in text or "apple tv plus" in text:
        return "Apple TV+"
    if "the tonight show" in text:
        return "NBC"
    if "late show" in text:
        return "CBS"
    if "late night with seth meyers" in text:
        return "NBC"
    if "saturday night live" in text:
        return "NBC"

    return ""


def infer_special_type(title: str = "", url: str = "", content: str = "") -> str:
    text = normalize_text(f"{title} {url} {content[:1500]}").lower()

    if any(x in text for x in ["oscars", "grammys", "golden globes", "emmys", "mark twain prize"]):
        return "Award Show"
    if any(x in text for x in ["the tonight show", "late night with", "late show", "saturday night live", "patriot act"]):
        return "TV Appearance"
    if "roast" in text:
        return "Roast"
    if "interview" in text:
        return "Interview"
    if "monologue" in text:
        return "Monologue"
    if "crowd work" in text:
        return "Crowd Work Special"

    return "Special"


# -----------------------------
# SENTENCE SPLITTING
# -----------------------------
def split_text_into_sentences(text: str) -> List[str]:
    text = normalize_for_structure(text)

    if not text:
        return []

    text = re.sub(r"\n{2,}", " <PARA> ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    rough_parts = re.split(r'(?<=[.!?])\s+|<PARA>', text)

    sentences: List[str] = []
    for part in rough_parts:
        part = re.sub(r"\s+", " ", part).strip()
        if not part:
            continue

        if len(part.split()) > 60:
            subparts = re.split(r'(?<=[;:])\s+', part)
            for sp in subparts:
                sp = sp.strip()
                if sp:
                    sentences.append(sp)
        else:
            sentences.append(part)

    return sentences


# -----------------------------
# TRANSCRIPT DOC BUILDING
# -----------------------------
def build_clean_transcript_docs(raw_transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []

    for doc_id, row in enumerate(raw_transcripts):
        url = row.get("url", "") or ""
        title = row.get("title", "") or ""
        raw_content = row.get("content", "") or ""

        content = clean_transcript_content(raw_content, title=title)
        content = normalize_text(content)

        comedian, special_title, release_date = parse_title_metadata(title)
        platform = infer_platform(url, title, content)
        special_type = infer_special_type(title, url, content)
        tokens = clean_and_tokenize_text(content)
        sentences = split_text_into_sentences(content)
        has_profanity, profanity_terms = detect_profanity(tokenize_for_flags(content))

        docs.append({
            "chunk_id": f"doc_{doc_id}",
            "doc_id": doc_id,
            "url": url,
            "title": title,
            "comedian": comedian,
            "special_title": special_title,
            "release_date": release_date,
            "platform": platform,
            "special_type": special_type,
            "content": content,
            "tokens": tokens,
            "length": len(tokens),
            "sentences": sentences,
            "has_profanity": has_profanity,
            "profanity_terms": profanity_terms,
        })

    return docs


# -----------------------------
# SEMANTIC CHUNKING
# -----------------------------
def mean_pool_and_normalize(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        raise ValueError("Cannot pool empty embeddings.")

    pooled = embeddings.mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm == 0:
        return pooled.astype(np.float32)
    return (pooled / norm).astype(np.float32)


def semantic_chunk_sentences(
    sentences: List[str],
    sentence_embeddings: np.ndarray,
    min_chunk_sentences: int = MIN_CHUNK_SENTENCES,
    min_chunk_words: int = MIN_CHUNK_WORDS,
    break_threshold: float = SEMANTIC_BREAK_THRESHOLD,
) -> Tuple[List[Tuple[List[str], np.ndarray, int, int]], List[float]]:
    if not sentences:
        return [], []

    if len(sentences) != len(sentence_embeddings):
        raise ValueError("Number of sentences and number of sentence embeddings must match.")

    if len(sentences) <= min_chunk_sentences:
        return [(sentences, sentence_embeddings, 0, len(sentences) - 1)], []

    chunks: List[Tuple[List[str], np.ndarray, int, int]] = []
    adjacent_similarities: List[float] = []

    current_sentences = [sentences[0]]
    current_embeddings = [sentence_embeddings[0]]
    current_start = 0

    for i in range(1, len(sentences)):
        sim_prev = float(np.dot(sentence_embeddings[i - 1], sentence_embeddings[i]))
        adjacent_similarities.append(sim_prev)

        current_word_count = count_words(" ".join(current_sentences))
        enough_sentences = len(current_sentences) >= min_chunk_sentences
        enough_words = current_word_count >= min_chunk_words

        if enough_sentences and enough_words and sim_prev < break_threshold:
            chunks.append((
                current_sentences,
                np.vstack(current_embeddings),
                current_start,
                i - 1,
            ))
            current_sentences = [sentences[i]]
            current_embeddings = [sentence_embeddings[i]]
            current_start = i
        else:
            current_sentences.append(sentences[i])
            current_embeddings.append(sentence_embeddings[i])

    if current_sentences:
        if chunks and (
            len(current_sentences) < min_chunk_sentences
            or count_words(" ".join(current_sentences)) < min_chunk_words
        ):
            prev_sentences, prev_embeddings, prev_start, _ = chunks[-1]
            merged_sentences = prev_sentences + current_sentences
            merged_embeddings = np.vstack([prev_embeddings, np.vstack(current_embeddings)])
            chunks[-1] = (merged_sentences, merged_embeddings, prev_start, len(sentences) - 1)
        else:
            chunks.append((
                current_sentences,
                np.vstack(current_embeddings),
                current_start,
                len(sentences) - 1,
            ))

    return chunks, adjacent_similarities


def build_chunks_from_docs(
    docs: List[Dict[str, Any]],
    model: SentenceTransformer,
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[float], Dict[int, List[int]]]:
    chunks: List[Dict[str, Any]] = []
    chunk_embeddings: List[np.ndarray] = []
    all_adjacent_similarities: List[float] = []
    transcript_chunk_ids: Dict[int, List[int]] = {}
    chunk_counter = 0

    for doc in docs:
        doc_id = doc["doc_id"]
        sentences = doc.get("sentences", [])

        transcript_chunk_ids[doc_id] = []

        if not sentences:
            continue

        sentence_embeddings = model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        sentence_chunks, adjacent_similarities = semantic_chunk_sentences(
            sentences=sentences,
            sentence_embeddings=sentence_embeddings,
            min_chunk_sentences=MIN_CHUNK_SENTENCES,
            min_chunk_words=MIN_CHUNK_WORDS,
            break_threshold=SEMANTIC_BREAK_THRESHOLD,
        )
        all_adjacent_similarities.extend(adjacent_similarities)

        for sent_chunk, sent_chunk_embeddings, start_idx, end_idx in sentence_chunks:
            chunk_text = " ".join(sent_chunk).strip()
            tokens = clean_and_tokenize_text(chunk_text)
            has_profanity, profanity_terms = detect_profanity(tokenize_for_flags(chunk_text))

            chunk = {
                "chunk_id": f"chunk_{chunk_counter}",
                "doc_id": doc_id,
                "comedian": doc.get("comedian", ""),
                "special_title": doc.get("special_title", ""),
                "release_date": doc.get("release_date", ""),
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "platform": doc.get("platform", ""),
                "special_type": doc.get("special_type", ""),
                "content": chunk_text,
                "tokens": tokens,
                "length": len(tokens),
                "chunk_sentences": sent_chunk,
                "sentence_start": start_idx,
                "sentence_end": end_idx,
                "global_snippet_start": start_idx,
                "global_snippet_end": end_idx,
                "has_profanity": has_profanity,
                "profanity_terms": profanity_terms,
            }

            chunks.append(chunk)
            chunk_embeddings.append(mean_pool_and_normalize(sent_chunk_embeddings))
            transcript_chunk_ids[doc_id].append(chunk_counter)
            chunk_counter += 1

    if chunk_embeddings:
        chunk_embedding_matrix = np.vstack(chunk_embeddings).astype(np.float32)
    else:
        chunk_embedding_matrix = np.zeros((0, 384), dtype=np.float32)

    return chunks, chunk_embedding_matrix, all_adjacent_similarities, transcript_chunk_ids


# -----------------------------
# DEBUG HELPERS
# -----------------------------
def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])

    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def print_adjacent_similarity_stats(adjacent_similarities: List[float]) -> None:
    if not adjacent_similarities:
        print("\nNo adjacent sentence similarity statistics available.")
        return

    values = sorted(adjacent_similarities)
    print("\nAdjacent sentence cosine similarity statistics:")
    print(f"  Count: {len(values)}")
    print(f"  Min: {values[0]:.4f}")
    print(f"  Max: {values[-1]:.4f}")
    print(f"  Mean: {statistics.mean(values):.4f}")
    print(f"  Median: {statistics.median(values):.4f}")
    print(f"  Std dev: {statistics.pstdev(values) if len(values) > 1 else 0.0:.4f}")
    print(f"  P10: {percentile(values, 0.10):.4f}")
    print(f"  P25: {percentile(values, 0.25):.4f}")
    print(f"  P50: {percentile(values, 0.50):.4f}")
    print(f"  P75: {percentile(values, 0.75):.4f}")
    print(f"  P90: {percentile(values, 0.90):.4f}")


# -----------------------------
# PAYLOAD BUILD
# -----------------------------
def build_chunk_index_payload(
    transcripts_path: str,
    save_debug_json: bool = DEFAULT_SAVE_DEBUG_JSON,
) -> Dict[str, Any]:
    raw_transcripts = load_raw_transcripts(transcripts_path)
    docs = build_clean_transcript_docs(raw_transcripts)

    model = get_embedding_model()
    chunks, chunk_embedding_matrix, adjacent_similarities, transcript_chunk_ids = build_chunks_from_docs(docs, model)

    if ENABLE_DEBUG_PRINTS:
        print_adjacent_similarity_stats(adjacent_similarities)

    return {
        "docs": docs,
        "chunks": chunks,
        "chunk_embedding_matrix": chunk_embedding_matrix,
        "transcript_chunk_ids": transcript_chunk_ids,
    }