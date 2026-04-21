"""
Uploads locally-built search index files to the GitHub Release so teammates
and the class server can download them via scripts/fetch_indices.py.

Run this after rebuilding indices locally (takes ~6 min):
  python scripts/upload_indices.py

Requires the GitHub CLI (gh) to be installed and authenticated:
  gh auth login

If the release does not exist yet it will be created automatically.
"""

import os
import subprocess
import sys

GITHUB_REPO = "ArielVilensky/laughDB"
RELEASE_TAG = "v1-indices"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "src", "data")

INDEX_FILES = [
    "transcript_docs.pkl",
    "transcript_search_index.pkl",
    "chunk_search_index.pkl",
]


def run(cmd: list[str]) -> int:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    return result.returncode


def main() -> None:
    missing = [f for f in INDEX_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"ERROR: Missing index files: {', '.join(missing)}", file=sys.stderr)
        print("Build them first by starting the app or running the index build script.", file=sys.stderr)
        sys.exit(1)

    print(f"Ensuring release '{RELEASE_TAG}' exists...")
    ret = run([
        "gh", "release", "create", RELEASE_TAG,
        "--repo", GITHUB_REPO,
        "--title", "Search Indices",
        "--notes", "Pre-built search index files.",
    ])
    if ret not in (0, 1):
        print("WARNING: Could not create release (may already exist — continuing).")

    failed = []
    for filename in INDEX_FILES:
        path = os.path.join(DATA_DIR, filename)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Uploading {filename} ({size_mb:.0f} MB) ...", flush=True)
        ret = run([
            "gh", "release", "upload", RELEASE_TAG,
            path,
            "--repo", GITHUB_REPO,
            "--clobber",
        ])
        if ret != 0:
            print(f"  ERROR: Failed to upload {filename}")
            failed.append(filename)
        else:
            print(f"  Done.")

    if failed:
        print(f"\nFailed to upload: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll {len(INDEX_FILES)} files uploaded to release '{RELEASE_TAG}'.")
    print(f"Teammates can now run: python scripts/fetch_indices.py")


if __name__ == "__main__":
    main()
