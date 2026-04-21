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
    "Lines 2+: 1 to 2 short paragraphs summarizing the results.\n\n"
    "For the VIBES line: choose exactly 3 comma-separated style descriptors that best capture "
    "the tone of these results. Pick from: dark, observational, absurdist, self-deprecating, "
    "political, crowd-work, storytelling, surreal, clean, edgy, dry, musical, topical, blue, "
    "cerebral, physical, deadpan, satirical, confessional.\n\n"
    "For the summary: highlight recurring themes, mention comedian names, note what kind of "
    "viewer would enjoy these. Be conversational, enthusiastic, and specific. Take into account "
    "result rank — higher rank means stronger match. Max 2 short paragraphs. No bullets or headers."
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
