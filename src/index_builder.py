import json
import os
import re
from typing import Tuple, List, Dict, Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PROFANITY_WORDS = {
    "fuck", "fucking", "fucked", "fucker",
    "shit", "shitty",
    "bitch", "bitches", "bitching",
    "asshole", "arsehole", "motherfucker", "dick", "ass"
    "pussy", "cunt", "cock", "cocksucker",
    "bastard", "damn", "nigger", "nigga", "coon",
    "kike", "chink", "spic", "beaner",
    "wanker", "twat", "blowjob", "anal"
}

_RE_REPEATED_LETTER = re.compile(r'([a-zA-Z])\1{2,}')
_RE_VOWEL = re.compile(r'[aeiouy]')
_RE_USERNAME = re.compile(r'[a-z]{3,}(\d{4,})')
_RE_YEAR = re.compile(r'(?:19|20)\d{2}')
_RE_MID_DIGIT = re.compile(r'[a-z]\d{3,}[a-z]')

def is_garbage_token(word: str) -> bool:
    if _RE_REPEATED_LETTER.search(word):
        return True
    if len(word) >= 5 and not word[0].isdigit() and not _RE_VOWEL.search(word.lower()):
        return True
    m = _RE_USERNAME.search(word.lower())
    if m and not _RE_YEAR.match(m.group(1)):
        return True
    if _RE_MID_DIGIT.search(word.lower()):
        return True
    return False


CUSTOM_STOPWORDS = {
    "im", "ive", "ill", "id",
    "youre", "youve", "youll", "youd",
    "were", "weve", "well", "wed",
    "shes","hes","yall",
    "theyre", "theyve", "theyll", "theyd",
    "thats", "theres", "whats", "wheres", "whos", "hows",
    "dont", "didnt", "doesnt", "isnt", "arent", "wasnt", "werent",
    "wouldnt", "couldnt", "shouldnt", "cant", "wont", "aint",
    "gonna", "wanna", "gotta",
    "uh", "um", "oh", "yeah", "hey", "ok", "okay", "joke", "jokes", "n",
    "ck","ckin","gil","legat","afrin","ni","ga","cos", "tch"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STREAMING_LINKS_PATH = os.path.join(DATA_DIR, "streaming_links.json")

DOC_COMEDIAN_OVERRIDES = {
    26: "Unknown",
    70: "Various Comedians",
    88: "John Oliver",
    89: "Bill Maher",
    239: "Jim Jefferies",
    315: "Unknown",
    392: "John Oliver",
    408: "John Oliver",
    409: "John Oliver",
    411: "John Oliver",
    413: "John Oliver",
    414: "John Oliver",
    415: "John Oliver",
    417: "John Oliver",
    418: "Seth Meyers",
    420: "Seth Meyers",
    421: "John Oliver",
    422: "John Oliver",
    445: "John Mulaney and Nick Kroll",
    464: "John Oliver",
    473: "John Oliver",
    488: "Larry the Cable Guy",
    509: "Seth Meyers",
}

def normalize_comedian_from_show_title(comedian: str, title: str) -> str:
    comedian_lc = str(comedian).lower().strip()
    title_lc = str(title).lower().strip()

    # Show-format rules applied regardless of whether comedian is already set
    rules = [
        ("last week tonight with john oliver", "John Oliver"),
        ("late night with seth meyers", "Seth Meyers"),
        ("real time with bill maher", "Bill Maher"),
        ("the jim jefferies show", "Jim Jefferies"),
        ("oh, hello on broadway", "John Mulaney and Nick Kroll"),
        ("miranda sings", "Colleen Ballinger"),
    ]
    for pattern, normalized_name in rules:
        if pattern in comedian_lc or pattern in title_lc:
            return normalized_name

    if comedian:
        return comedian

    # Title-only fallback rules (when comedian is empty)
    fallback_rules = [
        ("sincerely louis ck", "Louis C.K."),
        ("carlin at carnegie", "George Carlin"),
        ("saturday night news with george carlin", "George Carlin"),
        ("your friend, nate bargatze", "Nate Bargatze"),
        ("jimmy kimmel", "Jimmy Kimmel"),
        ("bill hicks at rodney", "Bill Hicks"),
    ]
    for pattern, normalized_name in fallback_rules:
        if pattern in title_lc:
            return normalized_name

    # Strip transcript suffixes and year for cleaner pattern matching
    clean = re.sub(r"\s*\[(?:full\s+)?transcript\].*$", "", title, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\s*[|–—]\s*(full\s+)?transcript.*$", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\s*\(\d{4}\).*$", "", clean).strip()
    clean = re.sub(r"\s*\[\d{1,2}/\d{1,2}/\d{2,4}\].*$", "", clean).strip()

    def _fix_case(name: str) -> str:
        return name.title() if name == name.upper() else name

    # Possessive: "Name's Title" → Name
    m = re.match(r"^([A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+){0,3})['\u2019][sS]\b", clean)
    if m:
        return _fix_case(m.group(1).strip())

    # "Name at/on Topic" — require 2+ words in name to avoid false positives
    m = re.match(r"^([A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+){1,3})\s+(?:[Aa]t|[Oo]n)\b", clean)
    if m:
        name = _fix_case(m.group(1).strip())
        if len(name.split()) >= 2:
            return name

    # "Name | Topic" — pipe separator
    m = re.match(r"^([A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+){0,3})\s*\|", clean)
    if m:
        return _fix_case(m.group(1).strip())

    # "Show with Name" — name at end, require 2+ words
    m = re.search(r"\bwith\s+([A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+){1,3})\s*$", clean)
    if m:
        name = _fix_case(m.group(1).strip())
        if len(name.split()) >= 2:
            return name

    return comedian

KNOWN_COMEDIAN_GROUPS = {
    frozenset(["john mulaney", "nick kroll"]): "John Mulaney and Nick Kroll",
}

def normalize_multi_comedian_name(comedian: str) -> str:
    if not comedian:
        return ""

    name = str(comedian).strip()

    # unify common separators
    name = re.sub(r"\s*(?:,|&|\+|/)\s*", " and ", name)

    parts = [part.strip() for part in name.split(" and ") if part.strip()]

    # remove duplicates while preserving order
    seen = set()
    unique_parts = []
    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            unique_parts.append(part)

    if not unique_parts:
        return ""

    if len(unique_parts) == 1:
        return unique_parts[0]

    return " and ".join(unique_parts)

def canonicalize_comedian_group(comedian: str) -> str:
    if not comedian:
        return ""

    parts = [part.strip().lower() for part in comedian.split(" and ") if part.strip()]
    if not parts:
        return comedian

    key = frozenset(parts)
    return KNOWN_COMEDIAN_GROUPS.get(key, comedian)

def normalize_comedian_name(comedian: str, title: str) -> str:
    comedian = normalize_comedian_from_show_title(comedian, title)
    comedian = normalize_multi_comedian_name(comedian)
    comedian = canonicalize_comedian_group(comedian)
    return comedian.strip()

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
                "yt": obj.get("yt", ""),
                "tags": obj.get("tags", []),
                "comedian": obj.get("comedian", ""),
                "year": obj.get("year", ""),
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
            "yt": obj.get("yt", ""),
            "tags": obj.get("tags", []),
            "comedian": obj.get("comedian", ""),
            "year": obj.get("year", ""),
        })
        idx = next_idx

    return transcripts


def load_streaming_links(path: str = STREAMING_LINKS_PATH) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}

    return data


def remove_bracketed_descriptions(text: str) -> str:
    if not text:
        return ""

    # Remove all square-bracket stage directions.
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Remove common parenthetical stage directions / sound cues.
    stage_direction_keywords = (
        r"cheering|applause|clapping|crowd|audience|laughing|laughter|laughs|"
        r"chuckling|chuckles|chattering|mumbling|music|playing|whistling|"
        r"announcer|narrator|intro|theme|floorboards creak|music stops|"
        r"music changes|pre-show music|crew|offstage"
    )

    text = re.sub(
        rf"\([^)]{{0,120}}\b(?:{stage_direction_keywords})\b[^)]{{0,120}}\)",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove ALL-CAPS inline sound cues that often survive without brackets.
    text = re.sub(
        r"\b(?:CHEERING(?: AND APPLAUSE)?|APPLAUSE|MUSIC STOPS|MUSIC CHANGES TO|"
        r"FLOORBOARDS CREAK|AUDIENCE CHEERS|AUDIENCE LAUGHING|AUDIENCE CLAPPING)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return text


def normalize_decades(text: str) -> str:
    text = re.sub(r"\b(\d{2})['’]s\b", r"\1s", text)
    text = re.sub(r"\b(\d{4})['’]s\b", r"\1s", text)
    return text

def remove_music_blocks(text: str) -> str:
    if not text:
        return ""

    # Remove bracketed music/song cues on their own line.
    text = re.sub(
        r"(?mi)^\s*\[[^\]\n]*(?:song|music|playing|starts?)\b[^\]\n]*\]\s*$",
        " ",
        text,
    )

    # Remove any remaining full lines that begin with musical notes.
    text = re.sub(
        r"(?mi)^\s*[♪♫].*$",
        " ",
        text,
    )

    # Remove inline note-wrapped lyric fragments.
    text = re.sub(
        r"[♪♫][^♪♫]{0,500}[♪♫]",
        " ",
        text,
    )

    # Remove stray note symbols.
    text = re.sub(r"[♪♫]+", " ", text)

    return text

def normalize_spaced_letters(text):
    """
    Example: "m i d g e t" --> "midget"
    Requires at least 3 single letters to avoid collapsing common 2-letter pairs like "I a" or "A B".
    """
    return re.sub(
        r'\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b',
        lambda match: re.sub(r'\s+', '', match.group(0)),
        text
    )

def normalize_text_helper(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"‘|’", "'", text)
    text = re.sub(r"“|”", '"', text)
    text = re.sub(r"\–|\—", "-", text)
    text = text.replace("\xa0", " ")
    return text

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = remove_music_blocks(text)
    text = remove_bracketed_descriptions(text)

    # Remove dangling bracket fragments left behind by partial earlier stripping,
    # e.g. "hing]" or "[camera".
    text = re.sub(r"\b[\w'\"-]{1,20}\]", " ", text)
    text = re.sub(r"\[[\w'\"-]{1,20}\b", " ", text)

    # Remove leftover sound/stage cue words that may survive bracket stripping.
    text = re.sub(
        r"\b(?:whooping|cheering|applause|music|band playing|camera shutter clicking|shutter clicking)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = normalize_text_helper(text)
    text = normalize_decades(text)
    text = normalize_spaced_letters(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_structure(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = normalize_text_helper(text)
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
        r"\bANNOUNCER:\b",
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
        r"\bYeah\b",
        r"\bWell\b",
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
        "premiered",
        "filmed at",
        "directed by",
        "executive produced",
        "transcript",
        "special",
        "comedian",
        "crowd work special",
        "netflix",
        "max",
        "amazon",
        "prime video",
        "stand-up comedy",
        "stand-up special",
        "aired on",
        "main segment",
        "other segments",
    ]

    if len(prefix.split()) >= 12 and any(sig in prefix_lc for sig in blurb_signals):
        return cleaned[earliest:].strip()

    return cleaned


def strip_episode_metadata_header(text: str) -> str:
    """Strip leading Season/Episode metadata blocks (LWT, SNL, Comedy Central, etc.)."""
    if not text:
        return text
    # Allow up to 30 chars of prefix (e.g. "Saturday Night Live ") before Season/Episode
    if not re.match(r'^.{0,30}(?:Season|Episode)\s*\d+', text, re.IGNORECASE):
        return text
    # LWT style: metadata ends before the host's standard opener
    lwt_opener = re.search(r'\b(Tonight[,\s]|Hi there[,!\s]|Welcome to\s)', text, re.IGNORECASE)
    if lwt_opener:
        return text[lwt_opener.start():]
    # Other Season/Episode cases (SNL, Comedy Central): strip just the label line and
    # let strip_leading_presenter_intro handle any Announcer prefix that follows
    label_m = re.match(
        r'^.{0,30}Season\s*\d+[,:]?\s*(?:Episode\s*\d+)?\s*',
        text, re.IGNORECASE,
    )
    if label_m:
        return text[label_m.end():]
    return text


def strip_leading_title_line(text: str) -> str:
    if not text:
        return ""
    return re.sub(
        r"^\s*[^.\n]{0,200}\|\s*Transcript\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def strip_leading_presenter_intro(text: str) -> str:
    import time

    patterns = [
        # Role labels
        r"^\s*(?:announcer|narrator|host|mc|voiceover)\s*:\s*",

        # Metadata / promo prefixes
        r"^\s*original air date\s*:\s*[^.?!\n]{0,250}",
        r"^\s*aired\s*[:\-]?\s*[^.?!\n]{0,250}",
        r"^\s*recorded on\s*[^.?!\n]{0,250}",
        r"^\s*recorded live(?: at)?\s*[^.?!\n]{0,250}",
        r"^\s*pre[- ]show music\s*:\s*[^.?!\n]{0,250}",
        r"^\s*this programme contains[^.?!\n]{0,250}",
        r"^\s*this program contains[^.?!\n]{0,250}",
        r"^\s*watch the full show(?: for free on youtube)?[^.?!\n]{0,250}",
        r"^\s*further reading\s*:\s*[^.?!\n]{0,400}",
        r"^\s*welcome to the[^.?!\n]{0,250}",
        r"^\s*welcome back to the[^.?!\n]{0,250}",

        # Presenter language
        r"^\s*ladies and gentlemen[,!:\-… ]*",
        r"^\s*would you please welcome[,!:\-… ]*",
        r"^\s*please welcome[,!:\-… ]*",
        r"^\s*let'?s bring (?:him|her|them) on(?: right now)?[,!:\-… ]*",
        r"^\s*give it up for[,!:\-… ]*",
        r"^\s*put your hands together for[,!:\-… ]*",
        r"^\s*coming to the stage(?: now)?[,!:\-… ]*",
        r"^\s*to the stage[,!:\-… ]*",
        r"^\s*here'?s[,!:\-… ]*",
        r"^\s*and now[,!:\-… ]*",

        # Bracketed starts / symbols
        r"^\s*\[audience [^\]]+\]\s*",
        r"^\s*\[music[^\]]*\]\s*",
        r"^\s*\[applause[^\]]*\]\s*",
        r"^\s*\[cheering[^\]]*\]\s*",
        r"^\s*\*+\s*",
        r"^\s*♪+\s*",
    ]

    current = text
    total_start = time.perf_counter()
    iteration = 0

    while True:
        iteration += 1
        changed = False
        before_iter = current

        for pattern in patterns:
            t0 = time.perf_counter()
            updated = re.sub(pattern, "", current, flags=re.IGNORECASE)
            dt = time.perf_counter() - t0

            if updated != current:
                current = updated
                changed = True

        # Remove leftover "Name:" or "Tom Papa," style fragments
        t0 = time.perf_counter()
        updated = re.sub(
            r"^\s*(?:mr\.?|mrs\.?|ms\.?|comedian\s+)?[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4}\s*[!,:.\-]+\s*",
            "",
            current,
        )
        dt = time.perf_counter() - t0

        if updated != current:
            current = updated
            changed = True

        if not changed:
            return current.strip()

        if current == before_iter:
            return current.strip()

def strip_transcript_promos(text: str) -> str:
    if not text:
        return ""

    working = text.strip()

    # Only inspect the beginning, because promo / metadata junk is a prefix problem.
    prefix_limit = min(len(working), 4000)
    prefix = working[:prefix_limit]
    prefix_lc = prefix.lower()

    promo_markers = [
        "transcript of",
        "now available",
        "[internal link]",
        "original air date",
        "aired",
        "recorded on",
        "recorded live at",
        "recorded live",
        "pre-show music",
        "watch the full show for free on youtube",
        "watch the full show",
        "further reading",
        "review by",
        "written and performed by",
        "original broadcast",
        "this programme contains",
        "this program contains",
        "welcome to the",
        "welcome back to the",
    ]

    speech_anchors = [
        "announcer:",
        "narrator:",
        "ladies and gentlemen",
        "would you please welcome",
        "please welcome",
        "give it up for",
        "to the stage",
        "thank you",
        "hello",
        "hi",
        "well",
        "yeah",
        "so",
        "how's everybody",
        "how are you",
    ]

    if not any(marker in prefix_lc for marker in promo_markers):
        return working

    anchor_positions = []
    for anchor in speech_anchors:
        pos = prefix_lc.find(anchor)
        if pos != -1:
            anchor_positions.append(pos)

    if anchor_positions:
        cut = min(anchor_positions)
        return working[cut:].strip()

    # If this is obviously just review / metadata junk and we never hit speech,
    # return empty so the doc gets skipped later.
    review_only_markers = [
        "further reading",
        "review by",
        "the a.v. club",
        "exclaim!",
        ".mp3",
        "original broadcast",
        "written and performed by",
    ]
    if any(marker in prefix_lc for marker in review_only_markers):
        return ""

    return working

def normalize_transcript_punctuation(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[\"]) +", r"\1", text)
    text = re.sub(r" +([)\]\"])", r"\1", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_transcript_content(text: str, title: str = "") -> str:
    import time

    if not text:
        return ""

    def _log(step_name: str, start_time: float, before_text: str, after_text: str) -> None:
        pass

    current = text

    t0 = time.perf_counter()
    before = current
    current = normalize_for_structure(current)
    _log("normalize_for_structure", t0, before, current)

    t0 = time.perf_counter()
    before = current
    current = strip_leading_title_line(current)
    _log("strip_leading_title_line", t0, before, current)

    t0 = time.perf_counter()
    before = current
    current = strip_episode_metadata_header(current)
    _log("strip_episode_metadata_header", t0, before, current)

    t0 = time.perf_counter()
    before = current
    current = strip_before_star_separator(current)
    _log("strip_before_star_separator", t0, before, current)

    # Run promo stripping early.
    t0 = time.perf_counter()
    before = current
    current = strip_transcript_promos(current)
    _log("strip_transcript_promos_1", t0, before, current)

    t0 = time.perf_counter()
    before = current
    current = strip_descriptive_blurb_before_speech(current)
    _log("strip_descriptive_blurb_before_speech", t0, before, current)

    # Run presenter stripping before normalization.
    t0 = time.perf_counter()
    before = current
    current = strip_leading_presenter_intro(current)
    _log("strip_leading_presenter_intro_1", t0, before, current)

    # Normalize text, including bracket/parenthetical cleanup.
    t0 = time.perf_counter()
    before = current
    current = normalize_text(current)
    _log("normalize_text", t0, before, current)

    # Run the prefix removers again after normalization, because
    # some intros become easier to match once bracket/music noise is removed.
    t0 = time.perf_counter()
    before = current
    current = strip_transcript_promos(current)
    _log("strip_transcript_promos_2", t0, before, current)

    t0 = time.perf_counter()
    before = current
    current = strip_leading_presenter_intro(current)
    _log("strip_leading_presenter_intro_2", t0, before, current)

    t0 = time.perf_counter()
    before = current
    current = normalize_transcript_punctuation(current)
    _log("normalize_transcript_punctuation", t0, before, current)

    return current.strip()


def tokenize_for_flags(text: str) -> List[str]:
    text = text.lower().replace("’", "'")
    return re.findall(r"[a-z0-9']+", text)


def clean_and_tokenize_text(text: str) -> List[str]:
    text = normalize_text(text).lower()
    words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)

    cleaned_tokens = []
    stopwords = set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS

    for token in words:
        collapsed = token.replace("'", "")
        if not collapsed:
            continue
        if collapsed in stopwords:
            continue
        cleaned_tokens.append(collapsed)

    return cleaned_tokens


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
    if "youtube" in haystack:
        return "YouTube"

    return ""


def infer_special_type(title: str, url: str, content: str) -> str:
    text = f"{title} {url} {content}".lower()

    if "crowd work" in text:
        return "Crowd Work Special"
    if "roast" in text:
        return "Roast"
    if "interview" in text:
        return "Interview"
    if any(x in text for x in ["monologue", "tonight show", "late show", "late night", "snl"]):
        return "Monologue"
    if any(x in text for x in ["oscars", "grammys", "golden globes", "emmys", "award"]):
        return "Award Show"
    if any(x in text for x in ["speech", "address"]):
        return "Speech"
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


def is_valid_http_url(value: str) -> bool:
    if not isinstance(value, str):
        return False
    value = value.strip()
    return bool(re.match(r"^https?://", value, flags=re.IGNORECASE))


def extract_tag_texts(tags: Any) -> List[str]:
    if not isinstance(tags, list):
        return []

    texts: List[str] = []
    for tag in tags:
        if isinstance(tag, dict):
            text = str(tag.get("text", "")).strip()
            if text:
                texts.append(text)
        elif isinstance(tag, str):
            text = tag.strip()
            if text:
                texts.append(text)
    return texts


def normalize_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clean_special_title_text(value: str) -> str:
    value = normalize_metadata_value(value)
    value = re.sub(r"\s*[|–-]\s*(full\s+)?transcript\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*\[(full\s+)?transcript\]\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" -|")
    return value


def looks_like_bad_comedian(value: str) -> bool:
    if not value:
        return True

    value_lc = value.lower().strip()

    bad_exact = {
        "stand-up transcripts",
        "full transcript",
        "transcript",
        "n/a",
        "na",
        "none",
        "unknown",
    }
    if value_lc in bad_exact:
        return True

    if re.search(r"\(\d{4}\)", value):
        return True
    if "transcript" in value_lc:
        return True
    if len(value.split()) > 6:
        return True

    return False


def choose_comedian(raw_comedian: str, parsed_comedian: str, tag_texts: List[str]) -> str:
    raw_comedian = normalize_metadata_value(raw_comedian)
    parsed_comedian = normalize_metadata_value(parsed_comedian)

    if raw_comedian and not looks_like_bad_comedian(raw_comedian):
        return raw_comedian

    for tag in tag_texts:
        if not looks_like_bad_comedian(tag):
            return tag

    return parsed_comedian


def normalize_year(raw_year: Any, parsed_year: str) -> str:
    raw_year_str = normalize_metadata_value(raw_year)
    if re.fullmatch(r"\d{4}", raw_year_str):
        return raw_year_str

    parsed_year = normalize_metadata_value(parsed_year)
    if re.fullmatch(r"\d{4}", parsed_year):
        return parsed_year

    return ""


def normalize_link_key_piece(text: str) -> str:
    text = normalize_metadata_value(text).lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*&\s*", " & ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_streaming_lookup_keys(comedian: str, special_title: str) -> List[str]:
    comedian_norm = normalize_link_key_piece(comedian)
    title_norm = normalize_link_key_piece(clean_special_title_text(special_title))

    if not comedian_norm or not title_norm:
        return []

    variants = {
        f"{comedian_norm}|{title_norm}",
        f"{comedian_norm}|{title_norm.replace('&', 'and')}",
        f"{comedian_norm}|{title_norm.replace('and', '&')}",
        f"{comedian_norm}|{title_norm.replace(' - ', ' – ')}",
        f"{comedian_norm}|{title_norm.replace(' – ', ' - ')}",
        f"{comedian_norm.replace(chr(34), '')}|{title_norm}",
    }

    return [v for v in variants if "|" in v]


def choose_watch_link(
    yt: str,
    comedian: str,
    special_title: str,
    streaming_links: Dict[str, Dict[str, str]],
) -> Tuple[str, str]:
    yt = normalize_metadata_value(yt)
    if is_valid_http_url(yt):
        return yt, "YouTube"

    for key in build_streaming_lookup_keys(comedian, special_title):
        entry = streaming_links.get(key)
        if isinstance(entry, dict):
            url = normalize_metadata_value(entry.get("url", ""))
            platform = normalize_metadata_value(entry.get("platform", ""))
            if is_valid_http_url(url):
                return url, platform or "Streaming"

    return "", ""


def build_transcript_index_payload(transcripts_path: str) -> Dict[str, Any]:
    raw_transcripts = load_raw_transcripts(transcripts_path)
    streaming_links = load_streaming_links()

    docs: List[Dict[str, Any]] = []
    skipped_empty = 0

    for doc_id, item in enumerate(raw_transcripts):
        url = item.get("url", "")
        title = item.get("title", "")
        raw_content = item.get("content", "")
        yt = normalize_metadata_value(item.get("yt", ""))
        raw_tags = item.get("tags", [])
        tag_texts = extract_tag_texts(raw_tags)

        parsed_comedian, parsed_special_title, parsed_release_date = parse_title_metadata(title)

        comedian = choose_comedian(item.get("comedian", ""), parsed_comedian, tag_texts)
        comedian = normalize_comedian_name(comedian, title)

        if not comedian:
            comedian = "Unknown"

        if doc_id in DOC_COMEDIAN_OVERRIDES:
            comedian = DOC_COMEDIAN_OVERRIDES[doc_id]

        special_title = clean_special_title_text(parsed_special_title or title)
        release_date = normalize_year(item.get("year", ""), parsed_release_date)

        watch_url, watch_platform = choose_watch_link(
            yt=yt,
            comedian=comedian,
            special_title=special_title,
            streaming_links=streaming_links,
        )

        cleaned_content = clean_transcript_content(raw_content, title=title)
        sentences = split_text_into_sentences(cleaned_content)

        if not cleaned_content or not sentences:
            skipped_empty += 1
            continue

        tokens = clean_and_tokenize_text(cleaned_content)
        sentence_tokens = [
            clean_and_tokenize_text(sentence)
            for sentence in sentences
        ]

        has_profanity, profanity_terms = detect_profanity(tokenize_for_flags(cleaned_content))
        inferred_platform = infer_platform(url, title, cleaned_content)
        platform = watch_platform or inferred_platform
        special_type = infer_special_type(title, url, cleaned_content)


        # -------------------------
        # BUILD DOC
        # -------------------------
        doc = {
            "doc_id": doc_id,
            "url": url,
            "yt": yt,
            "raw_tags": raw_tags,
            "tag_texts": tag_texts,
            "title": title,
            "comedian": comedian,
            "special_title": special_title,
            "release_date": release_date,
            "platform": platform,
            "watch_url": watch_url,
            "watch_platform": watch_platform,
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