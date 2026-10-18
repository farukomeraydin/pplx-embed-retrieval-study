### nfcorpus (3633 docs, 323 queries)

| system | nDCG@10 [95% CI] | recall@100 | index MB | CPU ms/query |
|---|---|---|---|---|
| BM25 | 0.307 [0.271, 0.343] | 0.240 | nan | nan |
| bge-small float32 | 0.344 [0.308, 0.380] | 0.311 | 5.6 | nan |
| bge-small int8 | 0.342 [0.306, 0.378] | 0.312 | 1.4 | nan |
| pplx-0.6b float (normalized) | 0.359 [0.324, 0.393] | 0.327 | 14.9 | 0.21 |
| pplx-0.6b binary | 0.327 [0.294, 0.362] | 0.283 | 0.5 | 1.33 |
| pplx-0.6b binary + float rescore (top-200) | 0.359 [0.324, 0.394] | 0.302 | 15.3 | nan |
| BM25 + pplx-0.6b (RRF) | 0.351 [0.315, 0.387] | 0.321 | nan | nan |

Paired nDCG@10 differences (CI excluding 0 => likely a real difference):
- pplx-0.6b float (normalized) minus bge-small float32: +0.015 [-0.003, +0.033]
- pplx-0.6b float (normalized) minus BM25: +0.052 [+0.029, +0.075]
- pplx-0.6b binary minus pplx-0.6b float (normalized): -0.032 [-0.043, -0.020]
- pplx-0.6b binary + float rescore (top-200) minus pplx-0.6b float (normalized): -0.000 [-0.005, +0.002]
- BM25 + pplx-0.6b (RRF) minus pplx-0.6b float (normalized): -0.008 [-0.024, +0.008]