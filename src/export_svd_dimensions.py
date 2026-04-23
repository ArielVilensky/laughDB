"""
Load the rebuilt search indices and write top-15 SVD dimension terms to
svd_dimensions.txt  (both transcript-level and chunk-level indices).
"""
import os, sys
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")

TRANSCRIPT_INDEX_PATH = os.path.join(DATA_DIR, "transcript_search_index.pkl")
CHUNK_INDEX_PATH      = os.path.join(DATA_DIR, "chunk_search_index.pkl")
OUT_PATH              = os.path.join(BASE_DIR, "..", "svd_dimensions.txt")

TOP_N = 20


def load_pickle(path: str):
    return joblib.load(path)


def write_index_section(fh, label: str, index: dict):
    dim_terms = index.get("dimension_terms", {})
    if not dim_terms:
        fh.write(f"  [no dimension_terms found in index]\n")
        return
    n_dims = len(dim_terms)
    fh.write(f"  {n_dims} dimensions\n\n")
    for dim_idx in sorted(dim_terms.keys()):
        pos = dim_terms[dim_idx].get("positive", [])[:TOP_N]
        neg = dim_terms[dim_idx].get("negative", [])[:TOP_N]
        fh.write(f"  Dimension {dim_idx}\n")
        fh.write(f"    (+) {', '.join(pos)}\n")
        fh.write(f"    (-) {', '.join(neg)}\n")
    fh.write("\n")


def main():
    lines = []
    with open(OUT_PATH, "w") as fh:
        fh.write("=" * 70 + "\n")
        fh.write(f"SVD DIMENSION TERMS  (top {TOP_N} per side, + and -)\n")
        fh.write("=" * 70 + "\n\n")

        for label, path in [
            ("TRANSCRIPT-LEVEL INDEX", TRANSCRIPT_INDEX_PATH),
            ("CHUNK-LEVEL INDEX",      CHUNK_INDEX_PATH),
        ]:
            fh.write(f"{'─'*70}\n{label}\n{'─'*70}\n")
            if not os.path.exists(path):
                fh.write("  [index file not found]\n\n")
                continue
            index = load_pickle(path)
            write_index_section(fh, label, index)

    print(f"Written → {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
