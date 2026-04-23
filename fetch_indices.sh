#!/usr/bin/env bash
set -e

BASE=https://github.com/ArielVilensky/laughDB/releases/download/v1-indices
DEST=src/data

mkdir -p "$DEST"

echo "Downloading search indices from GitHub Releases..."
curl -L --progress-bar -o "$DEST/transcript_docs.pkl"         "$BASE/transcript_docs.pkl"
curl -L --progress-bar -o "$DEST/transcript_search_index.pkl" "$BASE/transcript_search_index.pkl"
curl -L --progress-bar -o "$DEST/chunk_search_index.pkl"      "$BASE/chunk_search_index.pkl"
echo "Done."
