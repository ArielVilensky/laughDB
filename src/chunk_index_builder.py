import re
from collections import defaultdict
from typing import Tuple, List, Dict, Any, Optional

import numpy as np

from index_builder import (
    clean_and_tokenize_text,
    detect_profanity,
    tokenize_for_flags,
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_EMBEDDING_MODEL = None

MIN_CHUNK_SENTENCES = 3
MIN_CHUNK_WORDS = 80
SEMANTIC_BREAK_THRESHOLD = 0.14


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def semantic_chunk_sentences(
    sentences: List[str],
    sentence_embeddings: np.ndarray,
    min_chunk_sentences: int = MIN_CHUNK_SENTENCES,
    min_chunk_words: int = MIN_CHUNK_WORDS,
    break_threshold: float = SEMANTIC_BREAK_THRESHOLD,
) -> Tuple[List[Tuple[List[str], int, int]], List[float]]:
    if not sentences:
        return [], []

    if len(sentences) != len(sentence_embeddings):
        raise ValueError("Sentence count and embedding count do not match.")

    if len(sentences) <= min_chunk_sentences:
        return [(sentences, 0, len(sentences) - 1)], []

    chunks: List[Tuple[List[str], int, int]] = []
    adjacent_similarities: List[float] = []

    current_sentences = [sentences[0]]
    current_word_count = count_words(sentences[0])
    current_start = 0

    for i in range(1, len(sentences)):
        sim_prev = float(np.dot(sentence_embeddings[i - 1], sentence_embeddings[i]))
        adjacent_similarities.append(sim_prev)

        enough_sentences = len(current_sentences) >= min_chunk_sentences
        enough_words = current_word_count >= min_chunk_words

        if enough_sentences and enough_words and sim_prev < break_threshold:
            chunks.append((current_sentences, current_start, i - 1))
            current_sentences = [sentences[i]]
            current_word_count = count_words(sentences[i])
            current_start = i
        else:
            current_sentences.append(sentences[i])
            current_word_count += count_words(sentences[i])

    chunks.append((current_sentences, current_start, len(sentences) - 1))
    return chunks, adjacent_similarities


def merge_tiny_chunks(
    chunks: List[Tuple[List[str], int, int]],
    min_sentences: int = MIN_CHUNK_SENTENCES,
    min_words: int = MIN_CHUNK_WORDS,
) -> List[Tuple[List[str], int, int]]:
    if not chunks:
        return chunks

    merged = [chunks[0]]

    for sent_chunk, start_idx, end_idx in chunks[1:]:
        word_count = count_words(" ".join(sent_chunk))
        if len(sent_chunk) < min_sentences or word_count < min_words:
            prev_sent, prev_start, _ = merged[-1]
            merged[-1] = (prev_sent + sent_chunk, prev_start, end_idx)
        else:
            merged.append((sent_chunk, start_idx, end_idx))

    return merged


def build_semantic_chunks(
    docs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[float], Dict[int, List[int]]]:
    chunks: List[Dict[str, Any]] = []
    all_adjacent_similarities: List[float] = []
    transcript_chunk_ids: Dict[int, List[int]] = defaultdict(list)

    model = get_embedding_model()

    # Collect all sentences from all docs for a single encode call
    all_sentences: List[str] = []
    doc_sentence_ranges: List[Tuple[int, int]] = []
    offset = 0
    for doc in docs:
        sentences = doc.get("sentences", [])
        doc_sentence_ranges.append((offset, offset + len(sentences)))
        all_sentences.extend(sentences)
        offset += len(sentences)

    if all_sentences:
        all_embeddings = model.encode(
            all_sentences,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    else:
        all_embeddings = np.empty((0, model.get_sentence_embedding_dimension()))

    chunk_counter = 0

    for doc, (emb_start, emb_end) in zip(docs, doc_sentence_ranges):
        doc_id = doc["doc_id"]
        sentences = doc.get("sentences", [])
        sentence_tokens = doc.get("sentence_tokens", [])

        if not sentences:
            continue

        sentence_embeddings = all_embeddings[emb_start:emb_end]

        chunk_groups, adjacent_similarities = semantic_chunk_sentences(
            sentences=sentences,
            sentence_embeddings=sentence_embeddings,
        )
        chunk_groups = merge_tiny_chunks(chunk_groups)
        all_adjacent_similarities.extend(adjacent_similarities)

        for sent_chunk, start_idx, end_idx in chunk_groups:
            chunk_text = " ".join(sent_chunk).strip()
            tokens = clean_and_tokenize_text(chunk_text)
            has_profanity, profanity_terms = detect_profanity(tokenize_for_flags(chunk_text))

            chunk = {
                "chunk_id": f"chunk_{chunk_counter}",
                "doc_id": doc_id,
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "yt": doc.get("yt", ""),
                "raw_tags": doc.get("raw_tags", []),
                "tag_texts": doc.get("tag_texts", []),
                "comedian": doc.get("comedian", ""),
                "special_title": doc.get("special_title", ""),
                "release_date": doc.get("release_date", ""),
                "platform": doc.get("platform", ""),
                "watch_url": doc.get("watch_url", ""),
                "watch_platform": doc.get("watch_platform", ""),
                "special_type": doc.get("special_type", ""),
                "content": chunk_text,
                "tokens": tokens,
                "length": len(tokens),
                "chunk_sentences": sent_chunk,
                "chunk_sentence_tokens": sentence_tokens[start_idx:end_idx + 1],
                "sentence_start": start_idx,
                "sentence_end": end_idx,
                "global_snippet_start": start_idx,
                "global_snippet_end": end_idx,
                "has_profanity": has_profanity,
                "profanity_terms": profanity_terms,
            }

            chunks.append(chunk)
            transcript_chunk_ids[doc_id].append(chunk_counter)
            chunk_counter += 1

    return chunks, all_adjacent_similarities, transcript_chunk_ids
