"""
Downloads pre-built search index files from a GitHub Release if they are not
already present in the data/ directory. Run this once on the server before
starting the app to avoid rebuilding indices from scratch (~17 min).

Setup (do this once after building indices locally):
  1. Build indices locally by starting the app and letting it initialize.
  2. Create a GitHub release and upload the pkl files:
       gh release create v1-indices --title "Search Indices" --notes ""
       gh release upload v1-indices data/transcript_docs.pkl
       gh release upload v1-indices data/transcript_search_index.pkl
       gh release upload v1-indices data/chunk_search_index.pkl
  3. Update RELEASE_TAG below if you change the release.

Server usage:
  python scripts/fetch_indices.py

If the repo is private, set a GitHub personal access token:
  export GITHUB_TOKEN=your_token_here
"""

import os
import sys
import urllib.request

GITHUB_REPO = "ArielVilensky/laughDB"
RELEASE_TAG = "v1-indices"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "src", "data")

INDEX_FILES = [
    "transcript_docs.pkl",
    "transcript_search_index.pkl",
    "chunk_search_index.pkl",
]


def _make_opener() -> urllib.request.OpenerDirector:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return urllib.request.build_opener(
            urllib.request.HTTPSHandler(),
            urllib.request.HTTPBasicAuthHandler(),
        )
    return urllib.request.build_opener()


def download_file(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    token = os.environ.get("GITHUB_TOKEN")

    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"token {token}")

    print(f"  Downloading {os.path.basename(dest)} ...", flush=True)
    with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {pct:.0f}% ({downloaded // (1024*1024)} MB / {total // (1024*1024)} MB)", end="", flush=True)
    print()


def main() -> None:
    missing = [f for f in INDEX_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]

    if not missing:
        print("All index files already present.")
        return

    print(f"Missing {len(missing)} index file(s): {', '.join(missing)}")
    base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

    failed = []
    for filename in missing:
        url = f"{base_url}/{filename}"
        dest = os.path.join(DATA_DIR, filename)
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}")
            failed.append(filename)

    if failed:
        print(f"\nFailed to download: {', '.join(failed)}", file=sys.stderr)
        print("Check that RELEASE_TAG is correct and GITHUB_TOKEN is set for private repos.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Downloaded {len(missing) - len(failed)} file(s) to {DATA_DIR}/")


if __name__ == "__main__":
    main()
