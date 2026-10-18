import numpy as np


def ndcg_at_k(ranked, rels, k=10):
    gains = [rels.get(d, 0) for d in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, rels, k=100):
    rel = {d for d, r in rels.items() if r > 0}
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def evaluate(top_idx, data, k_ndcg=10, k_rec=100):
    """top_idx: (n_queries, K) doc indices, best first (negative = padding).
    Returns per-query score arrays so we can bootstrap later."""
    nd, rc = [], []
    for qi, qid in enumerate(data["qids"]):
        ranked = [data["doc_ids"][j] for j in top_idx[qi] if j >= 0]
        rels = data["qrels"][qid]
        nd.append(ndcg_at_k(ranked, rels, k_ndcg))
        rc.append(recall_at_k(ranked, rels, k_rec))
    return {f"ndcg@{k_ndcg}": np.array(nd), f"recall@{k_rec}": np.array(rc)}


def bootstrap_ci(x, n=2000, seed=0, alpha=0.05):
    """Mean and (1-alpha) percentile bootstrap CI over queries."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    means = x[idx].mean(axis=1)
    return x.mean(), np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)


def paired_diff_ci(a, b, n=2000, seed=0):
    """Mean per-query difference (a - b) with bootstrap CI. CI excluding 0 => likely real."""
    return bootstrap_ci(np.asarray(a) - np.asarray(b), n=n, seed=seed)
