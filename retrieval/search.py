import time
from collections import defaultdict

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- sparse ----------
def bm25_topk(docs, queries, k=100):
    import bm25s

    ret = bm25s.BM25()
    ret.index(bm25s.tokenize(docs, stopwords="en", show_progress=False), show_progress=False)
    res, _ = ret.retrieve(
        bm25s.tokenize(queries, stopwords="en", show_progress=False), k=k, show_progress=False
    )
    return np.asarray(res)


# ---------- vector prep / quantization ----------
def l2norm(x):
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)


def int8_scale(D_norm):
    """One global scale fitted on the (normalized) corpus, shared by queries."""
    return 127.0 / np.percentile(np.abs(D_norm), 99.9)


def to_int8(x_norm, scale):
    return np.clip(np.round(x_norm * scale), -127, 127).astype(np.int8)


def to_bits(x_norm, center):
    """1-bit quantization: sign of mean-centered vector."""
    return (x_norm - center) > 0


# ---------- exact dense search ----------
def dense_topk(Q, D, k=100, bs=256):
    D = torch.as_tensor(np.asarray(D), dtype=torch.float32, device=DEV)
    idx_all, sc_all = [], []
    for i in range(0, len(Q), bs):
        q = torch.as_tensor(np.asarray(Q[i : i + bs]), dtype=torch.float32, device=DEV)
        sc, ix = (q @ D.T).topk(min(k, D.shape[0]), dim=1)
        idx_all.append(ix.cpu().numpy())
        sc_all.append(sc.cpu().numpy())
    return np.concatenate(idx_all), np.concatenate(sc_all)


def binary_topk(Qb, Db, k=100, rescore=None, R=200, bs=256):
    """Hamming ranking (via +-1 dot products). If rescore=(Qfull, Dfull), take the top-R
    Hamming candidates and re-rank them with full-precision dot products."""
    Dpm = torch.as_tensor(Db, device=DEV).float() * 2 - 1
    if rescore is not None:
        Qf = torch.as_tensor(rescore[0], dtype=torch.float32, device=DEV)
        Df = torch.as_tensor(rescore[1], dtype=torch.float32, device=DEV)
    out = []
    for i in range(0, len(Qb), bs):
        qpm = torch.as_tensor(Qb[i : i + bs], device=DEV).float() * 2 - 1
        s = qpm @ Dpm.T
        if rescore is None:
            out.append(s.topk(min(k, s.shape[1]), dim=1)[1].cpu().numpy())
        else:
            cand = s.topk(min(R, s.shape[1]), dim=1)[1]  # (b, R)
            q = Qf[i : i + bs]
            fs = torch.einsum("bd,brd->br", q, Df[cand])
            order = fs.topk(min(k, fs.shape[1]), dim=1)[1]
            out.append(torch.gather(cand, 1, order).cpu().numpy())
    return np.concatenate(out)


# ---------- fusion / chunk aggregation ----------
def rrf(rank_lists, k=100, c=60):
    n = len(rank_lists[0])
    out = np.full((n, k), -1, dtype=np.int64)
    for qi in range(n):
        s = defaultdict(float)
        for rl in rank_lists:
            for r, d in enumerate(rl[qi]):
                s[int(d)] += 1.0 / (c + r + 1)
        top = sorted(s, key=s.get, reverse=True)[:k]
        out[qi, : len(top)] = top
    return out


def collapse_to_docs(chunk_idx, chunk2doc, k=100):
    """Max-pool chunk ranking into a doc ranking (first occurrence = best chunk)."""
    out = np.full((len(chunk_idx), k), -1, dtype=np.int64)
    for qi, row in enumerate(chunk_idx):
        seen, docs = set(), []
        for c in row:
            d = int(chunk2doc[c])
            if d not in seen:
                seen.add(d)
                docs.append(d)
                if len(docs) == k:
                    break
        out[qi, : len(docs)] = docs
    return out


# ---------- CPU latency micro-benchmarks (per query, single query at a time) ----------
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def bench_ms(fn, queries, n=200):
    qs = queries[: min(n, len(queries))]
    for q in qs[:5]:
        fn(q)
    t0 = time.perf_counter()
    for q in qs:
        fn(q)
    return (time.perf_counter() - t0) / len(qs) * 1000


def make_float_search(D, k=100):
    D = D.astype(np.float32)

    def f(q):
        s = D @ q
        i = np.argpartition(-s, k)[:k]
        return i[np.argsort(-s[i])]

    return f


def make_hamming_search(Dbits, k=100):
    Dp = np.packbits(Dbits, axis=1)

    def f(qbits):
        qp = np.packbits(qbits)
        dist = _POP[np.bitwise_xor(Dp, qp)].sum(axis=1, dtype=np.int32)
        i = np.argpartition(dist, k)[:k]
        return i[np.argsort(dist[i])]

    return f
