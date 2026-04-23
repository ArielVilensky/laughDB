"""
LLM summarization route — only loaded when USE_LLM = True in routes.py.
Adds POST /api/summarize that streams a brief AI summary of search results.
Supports an optional `followup` field for follow-up questions.
"""
import json
import os
import logging
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)

# Initial summary: asks for a VIBES prefix line then 1-2 paragraphs.
SYSTEM_PROMPT_SUMMARY = (
    "You are an enthusiastic comedy recommendation assistant for laughDB, a comedy transcript "
    "search engine. Respond in exactly this format — no deviation:\n\n"
    "Line 1: VIBES: [vibe1], [vibe2], [vibe3]\n"
    "Lines 2+: Exactly 1.5 paragraphs — a full first paragraph, then a short second paragraph of 1-2 sentences.\n\n"
    "For the VIBES line: choose exactly 3 comma-separated style descriptors that best capture "
    "the tone of these results. Pick from: dark, observational, absurdist, self-deprecating, "
    "political, crowd-work, storytelling, surreal, clean, edgy, dry, musical, topical, blue, "
    "cerebral, physical, deadpan, satirical, confessional.\n\n"
    "For the summary: highlight recurring themes, mention comedian names, note what kind of "
    "viewer would enjoy these. Be conversational, enthusiastic, and specific. Take into account "
    "result rank — higher rank means stronger match. First paragraph is 3-4 sentences. "
    "Second paragraph is 1-2 sentences max — a punchy closer. No bullets or headers."
)

# Follow-up: answer a specific question about the results in 1 short paragraph.
SYSTEM_PROMPT_FOLLOWUP = (
    "You are a comedy recommendation assistant for laughDB. Given the user's search context "
    "and a follow-up question, write a focused answer in exactly 1 short paragraph. "
    "Be specific — reference comedian names and details from the results. "
    "Do not start with phrases like 'Great question' or 'Certainly'. Just answer directly."
)


def _build_context(query: str, results: list) -> str:
    lines = []
    if query:
        lines.append(f'Search query: "{query}"\n')
    else:
        lines.append("User browsed without a specific query.\n")

    lines.append("Top matching comedy transcripts (ranked by relevance):\n")

    for r in results[:5]:
        rank = r.get("rank", "?")
        comedian = r.get("comedian") or "Unknown"
        title = r.get("special_title") or r.get("title") or ""
        date = r.get("release_date") or ""
        pct = r.get("similarity_percent")
        snippet = (r.get("display_snippet") or r.get("content") or "")[:280].strip()

        score_str = f" — {pct:.1f}% match" if pct is not None else ""
        date_str = f" ({date})" if date else ""
        lines.append(f"#{rank}: {comedian} — \"{title}\"{date_str}{score_str}")
        if snippet:
            lines.append(f'   Excerpt: "{snippet}"')
        lines.append("")

    return "\n".join(lines)


SYSTEM_PROMPT_BIT_TAGS = (
    "You are a comedy analysis assistant. Given a comedy transcript snippet, return ONLY a JSON array "
    "of exactly 2 short thematic tags (1-3 words each, title case) capturing the main themes. "
    "Examples: Relationships, Technology, Self-image, Ghosting, Class Anxiety, Social Media, Marriage, "
    "Race, Parenting, Politics, Aging, Food, Identity, Fame, Masculinity, etc.\n"
    'Return ONLY a JSON array. Example: ["Technology", "Relationships"]'
)

SYSTEM_PROMPT_BIT_FULL = (
    "You are a comedy analysis assistant for laughDB. Given a snippet from a stand-up comedy transcript, "
    "return ONLY a valid JSON object with exactly two fields:\n"
    '1. "tags": an array of exactly 2 short thematic tags (1-3 words each, title case).\n'
    '2. "summary": exactly 1-2 sentences analyzing what makes this bit work — the comedian\'s specific '
    "angle, the observation, or the construction. Be insightful and concise. No filler phrases.\n\n"
    'Return ONLY valid JSON. Example: {"tags": ["Technology", "Relationships"], '
    '"summary": "Ansari treats the phone as the defining object of modern romance."}'
)


def register_bit_summary_route(app):
    """Register the /api/bit-summary endpoint. Called from routes.py."""
    import json as json_mod

    @app.route("/api/bit-summary", methods=["POST"])
    def bit_summary():
        data = request.get_json() or {}
        snippet = (data.get("snippet") or "").strip()[:600]
        comedian = (data.get("comedian") or "").strip()
        special_title = (data.get("special_title") or "").strip()
        query = (data.get("query") or "").strip()

        if not snippet:
            return jsonify({"error": "No snippet provided"}), 400

        api_key = os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY not set"}), 500

        client = LLMClient(api_key=api_key)

        context_parts = []
        if comedian:
            context_parts.append(f"Comedian: {comedian}")
        if special_title:
            context_parts.append(f"Special: {special_title}")
        if query:
            context_parts.append(f"Search query that matched this: \"{query}\"")
        context_parts.append(f"Snippet: \"{snippet}\"")
        context = "\n".join(context_parts)

        tags_only = data.get("tags_only", False)
        system_prompt = SYSTEM_PROMPT_BIT_TAGS if tags_only else SYSTEM_PROMPT_BIT_FULL
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]

        try:
            full = ""
            for chunk in client.chat(messages, stream=True):
                if chunk.get("content"):
                    full += chunk["content"]

            tags_only = data.get("tags_only", False)
            if tags_only:
                # Response is a JSON array of tags
                start = full.find("[")
                end = full.rfind("]") + 1
                if start == -1 or end == 0:
                    return jsonify({"tags": [], "summary": ""})
                tags = json_mod.loads(full[start:end])
                return jsonify({"tags": tags if isinstance(tags, list) else [], "summary": ""})
            else:
                start = full.find("{")
                end = full.rfind("}") + 1
                if start == -1 or end == 0:
                    return jsonify({"error": "Could not parse AI response"}), 500
                result = json_mod.loads(full[start:end])
                return jsonify({
                    "tags": result.get("tags", []),
                    "summary": result.get("summary", ""),
                })
        except Exception as e:
            logger.error(f"bit-summary error: {e}")
            return jsonify({"error": "Failed to generate summary"}), 500


SYSTEM_PROMPT_QUERY_REFORMULATE = (
    "You are a search query optimizer for laughDB, a stand-up comedy transcript search engine "
    "with 500+ English-language specials from 1960–2025.\n\n"
    "Given the user's query, return ONLY a valid JSON object — no markdown, no explanation:\n"
    '{"query": "...", "comedian": null, "year_min": null, "year_max": null, "retrieval_hint": null, "note": null}\n\n'

    "=== QUERY field ===\n"
    "Rewrite as a keyword-rich IR string. Apply in order:\n"
    "1. Keep the original words as base. Remove only pure filler "
    "(find me, show me, I want, tell me about, what is, how do, etc.).\n"
    "2. Remove comedian names — put them in 'comedian'. Remove decade/year refs — put in year_min/year_max.\n"
    "3. If remaining query is already keyword-dense (≤5 words, no filler), keep all original words "
    "and add at most 1–2 tightly adjacent words a comedian would say on stage — the single most "
    "concrete related concept, not a synonym. Examples:\n"
    "   'old age' → 'old age wrinkles dying'  (adjacent concepts, not synonyms like 'elderly')\n"
    "   'sex jokes' → 'sex intimacy'  (keep 'sex', drop 'jokes' as filler, add 1 adjacent word)\n"
    "   'airplane food' → 'airplane food peanuts'  (one specific related word)\n"
    "   'Bill Burr marriage' → keep as-is  (already fully specific, nothing to add)\n"
    "4. Comedy STYLE words expand ONLY when the style word is the dominant intent of the entire "
    "query — not when it merely modifies a concrete topic noun. Examples:\n"
    "   'dark humor' → death tragedy morbid taboo shock  [style IS the query]\n"
    "   'dark family secrets' → keep as-is  [dark modifies a concrete topic]\n"
    "   'sex jokes' → sex  [topic is concrete; strip 'jokes' as mild filler, keep 'sex']\n"
    "   'clean comedy' → wholesome safe appropriate  [style IS the query]\n"
    "   'dry humor' → understated monotone awkward silence  [style IS the query]\n"
    "   'blue comedy' → sex crude explicit vulgar adult  [style IS the query]\n"
    "   'self-deprecating bits' → loser failure embarrassing awkward insecure  [style IS the query]\n"
    "   'absurdist stuff' → bizarre nonsense impossible weird  [style IS the query]\n"
    "5. Preserve slang and colloquial register — do NOT formalize "
    "(keep 'broke', NOT 'financially disadvantaged'; keep 'suck', NOT 'inadequate').\n"
    "6. If query is sarcastic or ironic ('how great it is to be broke', 'the joys of being fat'), "
    "flip the intent and search the satirized reality instead.\n"
    "7. For compound queries with AND/both ('comedy about family AND career'), "
    "include strong keywords for BOTH topics. Word limit is 4–10 words based on complexity.\n"
    "8. Use-case queries ('for my parents', 'for a work party', 'to show kids'): extract implied "
    "content constraints and search for those (e.g. clean wholesome safe).\n"
    "9. Referential queries about a named event ('OJ trial', '9/11', 'the slap'): keep specific "
    "event terms exactly — do not broaden to themes.\n\n"

    "=== COMEDIAN field ===\n"
    "Return the full name ONLY when the intent is to find THAT comedian's own specials — e.g. "
    "'Kevin Hart in his prime', 'early Pryor', 'best Chris Rock bits', 'Bill Burr marriage stuff'.\n"
    "Return null when the comedian is the SUBJECT of other people's material — e.g. 'jokes about Kevin Hart', "
    "'who roasts Trump', 'comedians talking about Kanye', 'what comedians say about LeBron'.\n"
    "Return null for comparisons ('X vs Y') or when no comedian is named.\n\n"

    "=== YEAR_MIN / YEAR_MAX fields ===\n"
    "Integer years ONLY if the user explicitly mentioned a decade, era, or career period. Examples:\n"
    "'90s' → 1990/1999. 'recent'/'modern' → 2015/null. 'classic'/'old school' → 1960/1990.\n"
    "'early Pryor' → 1968/1979. 'before the accident' (Pryor) → null/1980.\n"
    "IMPORTANT: Topic words like 'old', 'young', 'ancient', 'new' describing the SUBJECT of comedy "
    "(e.g. 'old age', 'young love') are NOT era references — set null.\n\n"

    "=== RETRIEVAL_HINT field ===\n"
    "'svd' — abstract, thematic, emotional, or exploratory (feelings, vibes, concepts, identity).\n"
    "'tfidf' — specific keywords, named people/events, catchphrases, quotes.\n"
    "null if unclear.\n\n"

    "=== NOTE field ===\n"
    "One short user-facing sentence if there is a known limitation. Cases:\n"
    "- Negation ('not political', 'without swearing', 'avoid'): keyword search cannot exclude topics.\n"
    "- Physical comedy ('slapstick', 'pratfall', 'facial expressions'): transcripts are text-only.\n"
    "- Craft/technique ('callbacks', 'misdirection', 'timing'): structure isn't in transcripts; "
    "searching by associated topics instead.\n"
    "- Comparison ('X vs Y'): suggest running two separate searches.\n"
    "- Very recent event (after 2024): may have limited corpus coverage.\n"
    "null otherwise."
)


def register_query_reformulate_route(app):
    """Register the /api/reformulate-query endpoint. Called from routes.py."""

    @app.route("/api/reformulate-query", methods=["POST"])
    def reformulate_query():
        data = request.get_json() or {}
        original = (data.get("query") or "").strip()
        if not original:
            return jsonify({"modified_query": "", "original_query": ""}), 400

        api_key = os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"modified_query": original, "original_query": original})

        client = LLMClient(api_key=api_key)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_QUERY_REFORMULATE},
            {"role": "user", "content": original},
        ]
        try:
            full = ""
            for chunk in client.chat(messages, stream=True):
                if chunk.get("content"):
                    full += chunk["content"]

            start = full.find("{")
            end = full.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON in response")

            parsed = json.loads(full[start:end])

            query = (parsed.get("query") or "").strip().strip('"').strip("'")
            comedian = ((parsed.get("comedian") or "").strip()) or None
            note = ((parsed.get("note") or "").strip()) or None
            retrieval_hint = (parsed.get("retrieval_hint") or "").strip().lower() or None
            if retrieval_hint not in {"tfidf", "svd"}:
                retrieval_hint = None

            def safe_year(val):
                try:
                    return int(val) if val is not None else None
                except (TypeError, ValueError):
                    return None

            year_min = safe_year(parsed.get("year_min"))
            year_max = safe_year(parsed.get("year_max"))

            return jsonify({
                "modified_query": query or original,
                "original_query": original,
                "comedian": comedian,
                "year_min": year_min,
                "year_max": year_max,
                "retrieval_hint": retrieval_hint,
                "note": note,
            })
        except Exception as e:
            logger.error(f"reformulate-query error: {e}")
            return jsonify({"modified_query": original, "original_query": original})


def register_chat_route(app, search_chunks):
    """Register the /api/summarize SSE endpoint. Called from routes.py."""

    @app.route("/api/summarize", methods=["POST"])
    def summarize():
        data = request.get_json() or {}
        query = (data.get("query") or "").strip()
        results = data.get("results") or []
        followup = (data.get("followup") or "").strip()

        if not results:
            return jsonify({"error": "No results to summarize"}), 400

        api_key = os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY not set — add it to your .env file"}), 500

        client = LLMClient(api_key=api_key)
        context = _build_context(query, results)

        if followup:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_FOLLOWUP},
                {"role": "user", "content": f"{context}\nFollow-up question: {followup}"},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
                {"role": "user", "content": context},
            ]

        def generate():
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
