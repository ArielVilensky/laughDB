"""
Routes: React app serving and transcript search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import os
from flask import send_from_directory, request, jsonify
from retrieval_updated import search_chunks, initialize_search

USE_LLM = False
# USE_LLM = True


def str_to_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes", "y", "on"}


def empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def optional_bool_arg(name: str, default: bool) -> bool:
    raw = request.args.get(name)
    if raw is None:
        return default
    return str_to_bool(raw)


def register_routes(app):
    # initialize_search()

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, 'index.html')

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": USE_LLM})

    @app.route("/api/search")
    def transcript_search():
        query = request.args.get("query", "").strip()

        if not query:
            return jsonify({
                "query": "",
                "results": [],
                "resolved_comedian": None,
                "known_comedians": [],
                "known_special_types": [],
            })

        top_k = request.args.get("top_k", default=25, type=int)
        top_k = clamp_int(top_k, 1, 25)

        retrieval_mode = request.args.get("retrieval_mode", "tfidf").strip().lower()
        if retrieval_mode not in {"tfidf", "svd", "embedding"}:
            retrieval_mode = "tfidf"

        result_scope = request.args.get("result_scope", "full").strip().lower()
        if result_scope not in {"chunks", "full"}:
            result_scope = "full"

        comedian = empty_to_none(request.args.get("comedian"))
        special_type = empty_to_none(request.args.get("special_type"))

        year_min = request.args.get("year_min", default=None, type=int)
        year_max = request.args.get("year_max", default=None, type=int)

        exclude_profanity = str_to_bool(request.args.get("exclude_profanity"))

        max_chunks_per_doc = request.args.get("max_chunks_per_doc", default=2, type=int)
        max_chunks_per_doc = clamp_int(max_chunks_per_doc, 1, 5)

        use_expensive_proximity_scoring = optional_bool_arg(
            "use_expensive_proximity_scoring",
            True,
        )
        show_svd_explanations = optional_bool_arg(
            "show_svd_explanations",
            True,
        )
        debug_score_breakdown = optional_bool_arg(
            "debug_score_breakdown",
            False,
        )

        results = search_chunks(
            query=query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            comedian=comedian,
            year_min=year_min,
            year_max=year_max,
            special_type=special_type,
            exclude_profanity=exclude_profanity,
            max_chunks_per_doc=max_chunks_per_doc,
            result_scope=result_scope,
            use_expensive_proximity_scoring=use_expensive_proximity_scoring,
            show_svd_explanations=show_svd_explanations,
            debug_score_breakdown=debug_score_breakdown,
        )
        return jsonify(results)

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, search_chunks)