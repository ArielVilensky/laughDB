import json
import re
from typing import Tuple, List, Dict, Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PROFANITY_WORDS = {
    "fuck", "fucking", "fucked", "fucker",
    "shit", "shitty",
    "bitch", "bitches", "bitching",
    "asshole", "motherfucker", "dick",
    "pussy", "cunt", "cock", "cocksucker",
    "bastard", "damn",
}

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
}


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


def remove_bracketed_descriptions(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(
        r"\([^)]{0,80}(cheering|applause|music|laughter|crowd|announcer|whistling)[^)]{0,80}\)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return text


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
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\xa0", " ")
    return text


def normalize_intro_separators(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*\s*\*\s*\*", "***", text)
    text = re.sub(r"\s+-\s+-\s+", " -- ", text)
    return text


def strip_before_star_separator(text: str) -> str:
    text = normalize_intro_separators(text)
    if "***" in text:
        parts = text.split("***", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
    return text.strip()


def strip_descriptive_blurb_before_speech(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()

    speech_markers = [
        r"\bAnnouncer:\b",
        r"\bLadies and gentlemen\b",
        r"\bThank you\b",
        r"\bHello\b",
        r"\bHi\b",
        r"\bSo\b",
        r"\bYou know\b",
        r"\bI probably\b",
        r"\bI am\b",
        r"\bI'm\b",
    ]

    earliest = None
    for pattern in speech_markers:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            if earliest is None or match.start() < earliest:
                earliest = match.start()

    if earliest is None:
        return cleaned

    prefix = cleaned[:earliest].strip()
    prefix_lc = prefix.lower()

    blurb_signals = [
        "release date:",
        "stars in",
        "directed by",
        "observations on",
        "is built around",
        "transcript",
        "special - an hour of stand-up",
        "comedian ",
    ]

    if len(prefix.split()) >= 12 and any(sig in prefix_lc for sig in blurb_signals):
        return cleaned[earliest:].strip()

    return cleaned


def strip_leading_title_line(text: str) -> str:
    if not text:
        return ""
    return re.sub(
        r"^\s*[^.\n]{0,160}\|\s*Transcript\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def strip_leading_presenter_intro(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()

    presenter_patterns = [
        r"^\s*\(?announcer\)?\s*:\s*",
        r"^\s*from\s+[A-Z][A-Za-z\s,'.-]{0,80},\s*",
        r"^\s*comedy central presents\b\s*",
        r"^\s*ladies and gentlemen[,!:\s-]*",
        r"^\s*please welcome[,!:\s-]*",
    ]

    changed = True
    while changed:
        changed = False
        for pattern in presenter_patterns:
            new_cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
            if new_cleaned != cleaned:
                cleaned = new_cleaned
                changed = True

        new_cleaned = re.sub(
            r"^\s*\(?announcer\)?\s*from\s+[^.?!]{0,120}?(comedy central presents|presents)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if new_cleaned != cleaned:
            cleaned = new_cleaned
            changed = True

    return cleaned


def strip_leading_stage_directions(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    patterns = [
        r"^\s*(\[[^\]]{1,120}\])\s*",
        r"^\s*(\((?:[^)]{0,120})\))\s*",
        r"^\s*(cheering|applause|crowd cheering|crowd cheering and whistling|classic rock music playing|music|laughter)\b[:\-]?\s*",
    ]

    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new_cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
            if new_cleaned != cleaned:
                cleaned = new_cleaned
                changed = True

    return cleaned


def clean_transcript_content(text: str, title: str = "") -> str:
    if not text:
        return ""

    working = normalize_for_structure(text)
    working = normalize_intro_separators(working)
    working = strip_before_star_separator(working)
    working = strip_descriptive_blurb_before_speech(working)
    working = strip_leading_title_line(working)
    working = strip_leading_presenter_intro(working)
    working = strip_leading_stage_directions(working)

    working = remove_bracketed_descriptions(working)
    working = re.sub(r"♪[^♪]*♪", " ", working)

    working = re.sub(
        r"^(ladies and gentlemen[^.?!]*[.?!]\s*)+",
        "",
        working.strip(),
        flags=re.IGNORECASE,
    )
    working = re.sub(
        r"^(please welcome[^.?!]*[.?!]\s*)+",
        "",
        working.strip(),
        flags=re.IGNORECASE,
    )

    working = normalize_transcript_punctuation(working)
    working = re.sub(r"\s+", " ", working).strip()
    return working


def tokenize_for_flags(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def clean_and_tokenize_text(text: str) -> List[str]:
    text = normalize_text(text).lower()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)

    filtered = []
    for tok in tokens:
        tok_compact = tok.replace("'", "")
        if tok_compact in ENGLISH_STOP_WORDS:
            continue
        if tok_compact in CUSTOM_STOPWORDS:
            continue
        filtered.append(tok_compact)

    return filtered


def detect_profanity(tokens: List[str]) -> Tuple[bool, List[str]]:
    found = sorted({tok for tok in tokens if tok in PROFANITY_WORDS})
    return (len(found) > 0, found)


def parse_title_metadata(title: str) -> Tuple[str, str, str]:
    title = title.strip()

    match = re.match(r"^(.*?):\s*(.*?)\s*\((\d{4})\)", title)
    if match:
        comedian = match.group(1).strip()
        special_title = match.group(2).strip()
        release_date = match.group(3).strip()
        return comedian, special_title, release_date

    match = re.match(r"^(.*?)\s*\((\d{4})\)", title)
    if match:
        return "", match.group(1).strip(), match.group(2).strip()

    return "", title, ""


def infer_platform(url: str, title: str, content: str) -> str:
    haystack = f"{url} {title} {content}".lower()

    if "netflix" in haystack:
        return "Netflix"
    if "hbo" in haystack or "max" in haystack:
        return "HBO"
    if "amazon" in haystack or "prime video" in haystack:
        return "Amazon Prime"
    if "comedy central" in haystack:
        return "Comedy Central"
    if "peacock" in haystack:
        return "Peacock"

    return ""


def infer_special_type(title: str, url: str, content: str) -> str:
    text = f"{title} {url} {content}".lower()

    if any(x in text for x in ["roast"]):
        return "Roast"
    if any(x in text for x in ["interview"]):
        return "Interview"
    if any(x in text for x in ["monologue", "tonight show", "late show", "late night", "snl"]):
        return "Monologue"
    if any(x in text for x in ["oscars", "grammys", "golden globes", "emmys", "award"]):
        return "Award Show"
    if any(x in text for x in ["speech", "address"]):
        return "Speech"
    if "crowd work" in text:
        return "Crowd Work Special"
    if any(x in text for x in ["special", "stand-up", "stand up"]):
        return "Special"
    if "tv appearance" in text:
        return "TV Appearance"

    return "Special"


def split_text_into_sentences(text: str) -> List[str]:
    text = normalize_for_structure(text)

    if not text:
        return []

    text = re.sub(r"\n{2,}", " <PARA> ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    protected = text
    protected = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.", r"\1<PERIOD>", protected)
    protected = re.sub(r"\b([A-Z])\.", r"\1<PERIOD>", protected)
    protected = re.sub(r"\b(U\.S|U\.K|L\.A|N\.Y)\.", lambda m: m.group(0).replace(".", "<PERIOD>"), protected)

    protected = re.sub(
        r"\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*:)",
        r" <SPLIT> \1",
        protected
    )

    protected = re.sub(
        r'([.!?]["\')\]]*)\s+(?=([A-Z0-9"\'(]|<PARA>|<SPLIT>))',
        r'\1 <SPLIT> ',
        protected
    )

    protected = protected.replace("<PARA>", " <SPLIT> ")

    parts = [p.strip() for p in protected.split("<SPLIT>") if p.strip()]

    sentences: List[str] = []
    for part in parts:
        part = part.replace("<PERIOD>", ".")
        part = re.sub(r"\s+", " ", part).strip()

        if not part:
            continue

        if sentences and len(part.split()) <= 2 and not re.search(r"[.!?]$", part):
            sentences[-1] = f"{sentences[-1]} {part}".strip()
            continue

        sentences.append(part)

    merged: List[str] = []
    for sent in sentences:
        if merged and re.match(r"^[a-z]", sent):
            merged[-1] = f"{merged[-1]} {sent}".strip()
        else:
            merged.append(sent)

    return merged


def normalize_transcript_punctuation(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[\"]) +", r"\1", text)
    text = re.sub(r" +([)\]\"])", r"\1", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_transcript_index_payload(transcripts_path: str) -> Dict[str, Any]:
    raw_transcripts = load_raw_transcripts(transcripts_path)
    docs: List[Dict[str, Any]] = []

    for doc_id, item in enumerate(raw_transcripts):
        url = item.get("url", "")
        title = item.get("title", "")
        raw_content = item.get("content", "")

        comedian, special_title, release_date = parse_title_metadata(title)
        cleaned_content = clean_transcript_content(raw_content, title=title)
        sentences = split_text_into_sentences(cleaned_content)

        if not cleaned_content or not sentences:
            continue

        tokens = clean_and_tokenize_text(cleaned_content)
        sentence_tokens = [clean_and_tokenize_text(sentence) for sentence in sentences]

        has_profanity, profanity_terms = detect_profanity(tokenize_for_flags(cleaned_content))
        platform = infer_platform(url, title, cleaned_content)
        special_type = infer_special_type(title, url, cleaned_content)

        doc = {
            "doc_id": doc_id,
            "url": url,
            "title": title,
            "comedian": comedian,
            "special_title": special_title,
            "release_date": release_date,
            "platform": platform,
            "special_type": special_type,
            "content": cleaned_content,
            "tokens": tokens,
            "length": len(tokens),
            "sentences": sentences,
            "sentence_tokens": sentence_tokens,
            "has_profanity": has_profanity,
            "profanity_terms": profanity_terms,
        }

        docs.append(doc)

    return {
        "docs": docs,
    }