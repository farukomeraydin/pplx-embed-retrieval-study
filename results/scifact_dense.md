### scifact (5183 docs, 300 queries)

| system | nDCG@10 [95% CI] | recall@100 | index MB | CPU ms/query |
|---|---|---|---|---|
| BM25 | 0.662 [0.614, 0.708] | 0.876 | nan | nan |
| bge-small float32 | 0.713 [0.669, 0.756] | 0.942 | 8.0 | nan |
| bge-small int8 | 0.708 [0.664, 0.751] | 0.940 | 2.0 | nan |
| pplx-0.6b float (normalized) | 0.726 [0.683, 0.769] | 0.960 | 21.2 | 0.22 |
| pplx-0.6b binary | 0.691 [0.647, 0.733] | 0.941 | 0.7 | 1.95 |
| pplx-0.6b binary + float rescore (top-200) | 0.726 [0.683, 0.769] | 0.947 | 21.9 | nan |
| BM25 + pplx-0.6b (RRF) | 0.714 [0.670, 0.757] | 0.959 | nan | nan |

Paired nDCG@10 differences (CI excluding 0 => likely a real difference):
- pplx-0.6b float (normalized) minus bge-small float32: +0.014 [-0.010, +0.036]
- pplx-0.6b float (normalized) minus BM25: +0.065 [+0.027, +0.104]
- pplx-0.6b binary minus pplx-0.6b float (normalized): -0.035 [-0.056, -0.016]
- pplx-0.6b binary + float rescore (top-200) minus pplx-0.6b float (normalized): +0.000 [+0.000, +0.000]
- BM25 + pplx-0.6b (RRF) minus pplx-0.6b float (normalized): -0.012 [-0.041, +0.017]