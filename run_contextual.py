"""Experiment 2: do contextual embeddings help chunk-level retrieval?

Each document is split into chunks of a few sentences. We rank chunks, then max-pool to docs.
  A) pplx-embed-v1 embeds every chunk independently
  B) pplx-embed-context-v1 embeds the chunks of a doc together (shared context)
  C) pplx-embed-v1 on the whole doc (reference)

python run_contextual.py --dataset scifact
"""
import argparse
import re
from pathlib import Path

import numpy as np

from retrieval import search as S
from retrieval.data import load_beir
from retrieval.embed import embed, embed_context, free_model
from retrieval.metrics import bootstrap_ci, evaluate, paired_diff_ci

STD = "perplexity-ai/pplx-embed-v1-0.6b"
CTX = "perplexity-ai/pplx-embed-context-v1-0.6b"


def chunk_doc(text, n=3):
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]
    chunks = [" ".join(sents[i : i + n]) for i in range(0, len(sents), n)]
    return chunks or [text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--sents", type=int, default=3, help="sentences per chunk")
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()
    ds, K = args.dataset, args.k

    data = load_beir(ds)
    chunked = [chunk_doc(d, args.sents) for d in data["docs"]]
    flat = [c for doc in chunked for c in doc]
    chunk2doc = np.array([i for i, doc in enumerate(chunked) for _ in doc])
    print(f"{len(data['docs'])} docs -> {len(flat)} chunks")
    tag = f"{ds}_s{args.sents}"

    runs = {}

    # C) whole-doc reference (shares the cache with run_dense.py)
    Dw = embed(STD, data["docs"], f"{ds}_docs")
    Q = embed(STD, data["queries"], f"{ds}_q")
    runs["standard, whole doc"] = S.dense_topk(S.l2norm(Q), S.l2norm(Dw), K)[0]

    # A) standard model on independent chunks
    Dc = embed(STD, flat, f"{tag}_chunks")
    free_model(STD)
    top = S.dense_topk(S.l2norm(Q), S.l2norm(Dc), 3 * K)[0]
    runs["standard, chunks (max-pool)"] = S.collapse_to_docs(top, chunk2doc, K)

    # B) contextual model: chunks embedded jointly per document; queries as 1-chunk docs
    Dx = embed_context(CTX, chunked, f"{tag}_ctx_chunks")
    Qx = embed_context(CTX, [[q] for q in data["queries"]], f"{ds}_ctx_q", docs_per_call=64)
    free_model(CTX)
    top = S.dense_topk(S.l2norm(Qx), S.l2norm(Dx), 3 * K)[0]
    runs["contextual, chunks (max-pool)"] = S.collapse_to_docs(top, chunk2doc, K)

    # B2) same contextual chunk vectors, but queries embedded by the standard model
    # (the model card recommends pplx-embed-v1 for queries). Tells us which query encoder fits.
    top = S.dense_topk(S.l2norm(Q), S.l2norm(Dx), 3 * K)[0]
    runs["contextual chunks, std-model queries (max-pool)"] = S.collapse_to_docs(top, chunk2doc, K)

    scores = {n: evaluate(i, data) for n, i in runs.items()}
    lines = [
        f"### contextual vs standard on {ds} ({args.sents} sentences/chunk)",
        "",
        "| system | nDCG@10 [95% CI] | recall@100 |",
        "|---|---|---|",
    ]
    for n, s in scores.items():
        m, lo, hi = bootstrap_ci(s["ndcg@10"])
        lines.append(f"| {n} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {s['recall@100'].mean():.3f} |")
    m, lo, hi = paired_diff_ci(
        scores["contextual, chunks (max-pool)"]["ndcg@10"],
        scores["standard, chunks (max-pool)"]["ndcg@10"],
    )
    lines += ["", f"contextual (ctx queries) minus standard (chunks), nDCG@10: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"]
    m, lo, hi = paired_diff_ci(
        scores["contextual chunks, std-model queries (max-pool)"]["ndcg@10"],
        scores["standard, chunks (max-pool)"]["ndcg@10"],
    )
    lines += [f"contextual (std queries) minus standard (chunks), nDCG@10: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"]

    # queries where each side wins most -> material for failure analysis
    a = scores["contextual, chunks (max-pool)"]["ndcg@10"]
    b = scores["standard, chunks (max-pool)"]["ndcg@10"]
    order = np.argsort(a - b)
    lines += ["", "Biggest wins for standard chunks (inspect these):"]
    for i in order[:5]:
        lines.append(f"- [{a[i] - b[i]:+.2f}] {data['queries'][i][:110]}")
    lines += ["", "Biggest wins for contextual chunks (inspect these):"]
    for i in order[::-1][:5]:
        lines.append(f"- [{a[i] - b[i]:+.2f}] {data['queries'][i][:110]}")

    report = "\n".join(lines)
    print("\n" + report)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{ds}_contextual.md").write_text(report)


if __name__ == "__main__":
    main()
