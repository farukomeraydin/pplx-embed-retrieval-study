"""Experiment 1: standard retrieval + quantization study.

python run_dense.py --dataset scifact
"""
import argparse
import json
from pathlib import Path

import numpy as np

from retrieval import search as S
from retrieval.data import load_beir
from retrieval.embed import embed, free_model
from retrieval.metrics import bootstrap_ci, evaluate, paired_diff_ci

PPLX = "perplexity-ai/pplx-embed-v1-0.6b"
BGE = "BAAI/bge-small-en-v1.5"
BGE_PROMPT = "Represent this sentence for searching relevant passages: "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")  # also try nfcorpus, fiqa
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args()

    ds, K = args.dataset, args.k
    data = load_beir(ds)
    print(f"{ds}: {len(data['docs'])} docs, {len(data['queries'])} queries")

    runs, size_mb, lat = {}, {}, {}

    # --- BM25 ---
    runs["BM25"] = S.bm25_topk(data["docs"], data["queries"], K)

    # --- small baseline dense model: float32 vs int8 ---
    D = embed(BGE, data["docs"], f"{ds}_docs", max_len=args.max_len)
    Q = embed(BGE, data["queries"], f"{ds}_q", prompt=BGE_PROMPT)
    free_model(BGE)
    Dn, Qn = S.l2norm(D), S.l2norm(Q)
    runs["bge-small float32"] = S.dense_topk(Qn, Dn, K)[0]
    sc = S.int8_scale(Dn)
    runs["bge-small int8"] = S.dense_topk(S.to_int8(Qn, sc), S.to_int8(Dn, sc), K)[0]
    size_mb["bge-small float32"] = Dn.nbytes / 1e6
    size_mb["bge-small int8"] = Dn.size / 1e6

    # --- pplx-embed-v1-0.6b: native vs binary vs binary+rescore ---
    D = embed(PPLX, data["docs"], f"{ds}_docs", max_len=args.max_len)
    Q = embed(PPLX, data["queries"], f"{ds}_q")
    free_model(PPLX)
    Dn, Qn = S.l2norm(D), S.l2norm(Q)
    center = Dn.mean(axis=0)
    Db, Qb = S.to_bits(Dn, center), S.to_bits(Qn, center)

    runs["pplx-0.6b float (normalized)"] = S.dense_topk(Qn, Dn, K)[0]
    runs["pplx-0.6b binary"] = S.binary_topk(Qb, Db, K)
    runs["pplx-0.6b binary + float rescore (top-200)"] = S.binary_topk(
        Qb, Db, K, rescore=(Qn, Dn), R=200
    )
    runs["BM25 + pplx-0.6b (RRF)"] = S.rrf([runs["BM25"], runs["pplx-0.6b float (normalized)"]], K)
    size_mb["pplx-0.6b float (normalized)"] = Dn.nbytes / 1e6
    size_mb["pplx-0.6b binary"] = Db.size / 8 / 1e6
    size_mb["pplx-0.6b binary + float rescore (top-200)"] = (Db.size / 8 + Dn.nbytes) / 1e6

    if not args.no_latency:
        lat["pplx-0.6b float (normalized)"] = S.bench_ms(S.make_float_search(Dn, K), Qn)
        lat["pplx-0.6b binary"] = S.bench_ms(S.make_hamming_search(Db, K), Qb)

    # --- evaluate ---
    scores = {name: evaluate(idx, data) for name, idx in runs.items()}
    lines = [
        f"### {ds} ({len(data['docs'])} docs, {len(data['queries'])} queries)",
        "",
        "| system | nDCG@10 [95% CI] | recall@100 | index MB | CPU ms/query |",
        "|---|---|---|---|---|",
    ]
    for name, s in scores.items():
        m, lo, hi = bootstrap_ci(s["ndcg@10"])
        lines.append(
            f"| {name} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {s['recall@100'].mean():.3f} | "
            f"{size_mb.get(name, float('nan')):.1f} | {lat.get(name, float('nan')):.2f} |"
        )

    def diff(a, b):
        m, lo, hi = paired_diff_ci(scores[a]["ndcg@10"], scores[b]["ndcg@10"])
        return f"- {a} minus {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    lines += ["", "Paired nDCG@10 differences (CI excluding 0 => likely a real difference):"]
    lines += [
        diff("pplx-0.6b float (normalized)", "bge-small float32"),
        diff("pplx-0.6b float (normalized)", "BM25"),
        diff("pplx-0.6b binary", "pplx-0.6b float (normalized)"),
        diff("pplx-0.6b binary + float rescore (top-200)", "pplx-0.6b float (normalized)"),
        diff("BM25 + pplx-0.6b (RRF)", "pplx-0.6b float (normalized)"),
    ]
    report = "\n".join(lines)
    print("\n" + report)

    Path("results").mkdir(exist_ok=True)
    Path(f"results/{ds}_dense.md").write_text(report)
    with open(f"results/{ds}_dense_perquery.json", "w") as f:
        json.dump({n: {k: v.tolist() for k, v in s.items()} for n, s in scores.items()}, f)


if __name__ == "__main__":
    main()
