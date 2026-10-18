### fiqa (57638 docs, 648 queries)

| system | nDCG@10 [95% CI] | recall@100 | index MB | CPU ms/query |
|---|---|---|---|---|
| BM25 | 0.233 [0.211, 0.257] | 0.515 | nan | nan |
| bge-small float32 | 0.403 [0.375, 0.431] | 0.696 | 88.5 | nan |
| bge-small int8 | 0.401 [0.373, 0.429] | 0.688 | 22.1 | nan |
| pplx-0.6b float (normalized) | 0.519 [0.491, 0.549] | 0.828 | 236.1 | 7.84 |
| pplx-0.6b binary | 0.494 [0.466, 0.524] | 0.806 | 7.4 | 23.03 |
| pplx-0.6b binary + float rescore (top-200) | 0.519 [0.491, 0.549] | 0.825 | 243.5 | nan |
| BM25 + pplx-0.6b (RRF) | 0.394 [0.368, 0.421] | 0.799 | nan | nan |

Paired nDCG@10 differences (CI excluding 0 => likely a real difference):
- pplx-0.6b float (normalized) minus bge-small float32: +0.116 [+0.096, +0.135]
- pplx-0.6b float (normalized) minus BM25: +0.287 [+0.259, +0.314]
- pplx-0.6b binary minus pplx-0.6b float (normalized): -0.025 [-0.036, -0.015]
- pplx-0.6b binary + float rescore (top-200) minus pplx-0.6b float (normalized): +0.000 [+0.000, +0.000]
- BM25 + pplx-0.6b (RRF) minus pplx-0.6b float (normalized): -0.125 [-0.147, -0.105]