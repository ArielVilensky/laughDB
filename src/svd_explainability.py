from typing import Any, Dict, List

import numpy as np



def get_top_query_dimensions(query_latent: np.ndarray, top_n: int = 8) -> List[Dict[str, Any]]:
    ranked = sorted(
        enumerate(query_latent),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    return [
        {
            "dimension": int(dim_idx),
            "query_value": float(value),
            "sign": "positive" if value >= 0 else "negative",
        }
        for dim_idx, value in ranked
    ]



def get_top_dimension_contributions(
    query_latent: np.ndarray,
    doc_latent: np.ndarray,
    top_n: int = 8,
) -> List[Dict[str, Any]]:
    contributions = query_latent * doc_latent
    ranked = sorted(
        enumerate(contributions),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    results: List[Dict[str, Any]] = []
    for dim_idx, value in ranked:
        results.append(
            {
                "dimension": int(dim_idx),
                "contribution": float(value),
                "query_value": float(query_latent[dim_idx]),
                "doc_value": float(doc_latent[dim_idx]),
                "sign": "positive" if value >= 0 else "negative",
            }
        )
    return results



def describe_svd_dimension(
    components: np.ndarray,
    vocab: List[str],
    dim_idx: int,
    top_n: int = 10,
) -> Dict[str, Any]:
    component = components[dim_idx]
    ranked = np.argsort(component)

    top_negative = [vocab[i] for i in ranked[:top_n]]
    top_positive = [vocab[i] for i in ranked[::-1][:top_n]]

    return {
        "dimension": int(dim_idx),
        "positive_terms": top_positive,
        "negative_terms": top_negative,
    }



def summarize_svd_space(
    explained_variance_ratio: np.ndarray,
    n_dims: int = 10,
) -> Dict[str, Any]:
    n_dims = min(n_dims, len(explained_variance_ratio))
    per_dim = [float(x) for x in explained_variance_ratio[:n_dims]]
    return {
        "top_dimensions_variance": per_dim,
        "cumulative_variance": float(np.sum(explained_variance_ratio[:n_dims])),
        "total_dimensions": int(len(explained_variance_ratio)),
    }
