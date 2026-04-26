"""
Routes: React app serving and transcript search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
from flask import send_from_directory, request, jsonify
from retrieval_updated import search_chunks, initialize_search

_CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "search_cache.db")
_CACHE_MAX_ENTRIES = 500
_cache_lock = threading.Lock()
_cache_conn: sqlite3.Connection | None = None


def _get_cache_conn() -> sqlite3.Connection:
    global _cache_conn
    if _cache_conn is None:
        os.makedirs(os.path.dirname(_CACHE_DB_PATH), exist_ok=True)
        _cache_conn = sqlite3.connect(_CACHE_DB_PATH, check_same_thread=False)
        _cache_conn.execute(
            "CREATE TABLE IF NOT EXISTS search_cache "
            "(key TEXT PRIMARY KEY, result_json TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        _cache_conn.commit()
    return _cache_conn


def _cache_key(*args) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()


def _cache_get(key: str):
    try:
        row = _get_cache_conn().execute(
            "SELECT result_json FROM search_cache WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def _cache_set(key: str, result) -> None:
    try:
        with _cache_lock:
            conn = _get_cache_conn()
            conn.execute(
                "INSERT OR REPLACE INTO search_cache (key, result_json, created_at) VALUES (?,?,?)",
                (key, json.dumps(result), time.time()),
            )
            count = conn.execute("SELECT COUNT(*) FROM search_cache").fetchone()[0]
            if count > _CACHE_MAX_ENTRIES:
                conn.execute(
                    "DELETE FROM search_cache WHERE key IN "
                    "(SELECT key FROM search_cache ORDER BY created_at ASC LIMIT ?)",
                    (count - _CACHE_MAX_ENTRIES,),
                )
            conn.commit()
    except Exception:
        pass

# USE_LLM = False
USE_LLM = True

DEFAULT_YEAR_MIN = 1965
DEFAULT_YEAR_MAX = 2026
DEFAULT_RETRIEVAL_MODE = "tfidf"
DEFAULT_RESULT_SCOPE = "full"


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

    @app.route("/api/comedians")
    def comedians():
        from retrieval_updated import get_transcript_index, get_known_comedians
        idx = get_transcript_index()
        names = get_known_comedians(idx.get("items", []))
        return jsonify(names)

    @app.route("/api/search")
    def transcript_search():
        query = request.args.get("query", "").strip()

        comedian = empty_to_none(request.args.get("comedian"))
        special_type = empty_to_none(request.args.get("special_type"))
        year_min = request.args.get("year_min", default=None, type=int)
        year_max = request.args.get("year_max", default=None, type=int)
        exclude_profanity = str_to_bool(request.args.get("exclude_profanity"))

        top_k = request.args.get("top_k", default=25, type=int)
        top_k = clamp_int(top_k, 1, 25)

        retrieval_mode = request.args.get("retrieval_mode", DEFAULT_RETRIEVAL_MODE).strip().lower()
        if retrieval_mode not in {"tfidf", "svd"}:
            retrieval_mode = DEFAULT_RETRIEVAL_MODE

        result_scope = request.args.get("result_scope", DEFAULT_RESULT_SCOPE).strip().lower()
        if result_scope not in {"chunks", "full"}:
            result_scope = DEFAULT_RESULT_SCOPE

        has_active_filters = any([
            bool(comedian),
            bool(special_type),
            exclude_profanity,
            year_min is not None and year_min != DEFAULT_YEAR_MIN,
            year_max is not None and year_max != DEFAULT_YEAR_MAX,
            retrieval_mode != DEFAULT_RETRIEVAL_MODE,
            result_scope != DEFAULT_RESULT_SCOPE,
        ])

        if not query and not has_active_filters:
            return jsonify({
                "query": "",
                "results": [],
                "resolved_comedian": None,
                "known_comedians": [],
                "known_special_types": [],
            })

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

        ck = _cache_key(
            query, result_scope, retrieval_mode, comedian,
            year_min, year_max, special_type, exclude_profanity, max_chunks_per_doc,
        )
        cached = _cache_get(ck)
        if cached is not None:
            return jsonify(cached)

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
        )
        _cache_set(ck, results)
        return jsonify(results)

    if USE_LLM:
        from llm_routes import register_chat_route, register_bit_summary_route, register_query_reformulate_route
        register_chat_route(app, search_chunks)
        register_bit_summary_route(app)
        register_query_reformulate_route(app)