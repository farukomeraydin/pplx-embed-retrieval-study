### contextual vs standard on nfcorpus (3 sentences/chunk)

| system | nDCG@10 [95% CI] | recall@100 |
|---|---|---|
| standard, whole doc | 0.359 [0.324, 0.393] | 0.327 |
| standard, chunks (max-pool) | 0.345 [0.309, 0.380] | 0.309 |
| contextual, chunks (max-pool) | 0.347 [0.310, 0.382] | 0.324 |
| contextual chunks, std-model queries (max-pool) | 0.326 [0.293, 0.360] | 0.300 |

contextual (ctx queries) minus standard (chunks), nDCG@10: +0.002 [-0.011, +0.016]
contextual (std queries) minus standard (chunks), nDCG@10: -0.018 [-0.032, -0.006]

Biggest wins for standard chunks (inspect these):
- [-0.63] Vitamin C-Enriched Bacon
- [-0.51] walnut oil
- [-0.47] Is Coconut Milk Good For You?
- [-0.35] grapes
- [-0.33] Meat & Multiple Myeloma

Biggest wins for contextual chunks (inspect these):
- [+0.45] BMAA
- [+0.40] blood clots
- [+0.39] Eating Healthy on a Budget
- [+0.39] Panama
- [+0.36] Atkins diet