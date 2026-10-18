# pplx-embed retrieval study

A small, reproducible benchmark of Perplexity's open-source [`pplx-embed`](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) embedding models on three BEIR datasets. It asks three practical questions:

1. **Quantization:** how much retrieval quality do 1-bit embeddings cost, how much index size do they save, and does float rescoring of the top candidates recover the loss?
2. **Contextual embeddings:** does embedding the chunks of a document *together* (`pplx-embed-context-v1`) beat embedding them independently (`pplx-embed-v1`)?
3. **Hybrid search:** does fusing BM25 with dense retrieval beat dense retrieval alone?

Every number comes with a 95% bootstrap confidence interval over queries, and system comparisons use paired bootstrap differences, so "better" means "the interval excludes zero", not "the point estimate is higher".

> Independent project. Not affiliated with Perplexity.

## Summary

- **1-bit quantization shrinks the index ~30x for a small, real quality loss** (about -0.03 nDCG@10). Rescoring the top 200 binary candidates with float vectors recovers nDCG@10 completely on all three datasets.
- **Contextual embeddings gave no consistent gain.** The effect changes sign depending on the dataset and on which model embeds the queries.
- **`pplx-embed-0.6b` clearly beats BM25 everywhere, but beats `bge-small` only on FiQA** (+0.116). On SciFact and NFCorpus the difference is within noise.
- **Naive equal-weight RRF with BM25 can hurt:** -0.125 nDCG@10 on FiQA, where BM25 is far weaker than the dense model.

## Setup

Tested on Windows with an NVIDIA RTX 3080 Ti (12 GB). A CUDA build of PyTorch is strongly recommended (the code prints which device it loads each model on).

```bash
pip install -r requirements.txt

python run_dense.py --dataset scifact       # quantization, hybrid, baselines
python run_contextual.py --dataset scifact  # contextual vs standard embeddings
# repeat with: nfcorpus, fiqa
```

Datasets download automatically from the public BEIR host into `data/`. Embeddings are cached in `cache/` (checkpointed in blocks, so an interrupted run resumes). Reports are written to `results/`.

## Results

All numbers are on the BEIR test split. nDCG@10 with 95% bootstrap CI.

### Retrieval and quantization

| system | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| BM25 | 0.662 [0.614, 0.708] | 0.307 [0.271, 0.343] | 0.233 [0.211, 0.257] |
| bge-small float32 | 0.713 [0.669, 0.756] | 0.344 [0.308, 0.380] | 0.403 [0.375, 0.431] |
| bge-small int8 | 0.708 [0.664, 0.751] | 0.342 [0.306, 0.378] | 0.401 [0.373, 0.429] |
| pplx-0.6b float | 0.726 [0.683, 0.769] | 0.359 [0.324, 0.393] | 0.519 [0.491, 0.549] |
| pplx-0.6b binary | 0.691 [0.647, 0.733] | 0.327 [0.294, 0.362] | 0.494 [0.466, 0.524] |
| pplx-0.6b binary + float rescore (top-200) | 0.726 [0.683, 0.769] | 0.359 [0.324, 0.394] | 0.519 [0.491, 0.549] |
| BM25 + pplx-0.6b (RRF) | 0.714 [0.670, 0.757] | 0.351 [0.315, 0.387] | 0.394 [0.368, 0.421] |

Paired nDCG@10 differences (a CI that excludes 0 suggests a real difference):

| comparison | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| pplx float − bge-small float32 | +0.014 [−0.010, +0.036] | +0.015 [−0.003, +0.033] | **+0.116 [+0.096, +0.135]** |
| pplx float − BM25 | **+0.065 [+0.027, +0.104]** | **+0.052 [+0.029, +0.075]** | **+0.287 [+0.259, +0.314]** |
| pplx binary − pplx float | **−0.035 [−0.056, −0.016]** | **−0.032 [−0.043, −0.020]** | **−0.025 [−0.036, −0.015]** |
| pplx binary + rescore − pplx float | +0.000 [+0.000, +0.000] | −0.000 [−0.005, +0.002] | +0.000 [+0.000, +0.000] |
| BM25 + pplx (RRF) − pplx float | −0.012 [−0.041, +0.017] | −0.008 [−0.024, +0.008] | **−0.125 [−0.147, −0.105]** |

### Index size and recall (pplx-0.6b)

| | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| float index (MB) | 21.2 | 14.9 | 236.1 |
| binary index (MB) | 0.7 | 0.5 | 7.4 |
| recall@100, float | 0.960 | 0.327 | 0.828 |
| recall@100, binary + rescore | 0.947 | 0.302 | 0.825 |

Rescoring matches float nDCG@10, but recall@100 is slightly lower because the candidate pool is capped at 200 before re-ranking. The memory saving also assumes the float vectors can sit on slower storage (the "binary + rescore" index still needs them).

### Contextual vs standard embeddings (chunk-level retrieval, 3 sentences per chunk, max-pooled to documents)

| system | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| standard, whole document | 0.726 [0.683, 0.769] | 0.359 [0.324, 0.393] | 0.519 [0.491, 0.549] |
| standard, chunks | 0.736 [0.694, 0.778] | 0.345 [0.309, 0.380] | 0.441 [0.413, 0.470] |
| contextual chunks, queries from contextual model | 0.684 [0.638, 0.731] | 0.347 [0.310, 0.382] | 0.454 [0.426, 0.484] |
| contextual chunks, queries from standard model | 0.741 [0.698, 0.782] | 0.326 [0.293, 0.360] | 0.474 [0.445, 0.502] |

Paired difference vs "standard, chunks":

| contextual variant | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| queries from contextual model | **−0.053 [−0.081, −0.026]** | +0.002 [−0.011, +0.016] | +0.013 [−0.006, +0.031] |
| queries from standard model | +0.004 [−0.018, +0.026] | **−0.018 [−0.032, −0.006]** | **+0.033 [+0.015, +0.049]** |

Full tables and per-query scores are in [`results/`](results/).

## Findings

- **Quantization is a good trade.** Binary vectors cost 0.025–0.035 nDCG@10 and cut the index about 30x. A two-stage setup (binary candidates, then float rescoring of the top 200) matched float nDCG@10 on every dataset.
- **Contextual embeddings did not help consistently.** With the contextual model embedding the queries, SciFact got significantly worse and the other two datasets were flat. With the standard model embedding the queries, FiQA improved significantly (+0.033) but NFCorpus got significantly worse. The choice of query encoder mattered as much as the choice of document encoder.
- **Chunking itself can cost more than contextual embedding recovers.** On FiQA and NFCorpus, embedding the whole document beat every chunked variant (FiQA: 0.519 vs 0.474 at best). BEIR labels relevance per document, which favors whole-document embedding, so this says little about long documents where whole-document embedding isn't feasible.
- **The dense model's advantage depends on the dataset.** It beats BM25 everywhere, but beats `bge-small` significantly only on FiQA.
- **Hybrid fusion is not free.** Unweighted RRF with a much weaker BM25 pulled FiQA down significantly, and was neutral on the other two datasets.

## Method

- **Data:** BEIR SciFact, NFCorpus and FiQA, test split. Documents are `title + "\n" + text`. A few FiQA documents are empty and are replaced with a placeholder (`"N/A"`), because the model produces zero-length sequences for empty strings.
- **Models:** `perplexity-ai/pplx-embed-v1-0.6b`, `perplexity-ai/pplx-embed-context-v1-0.6b`, and `BAAI/bge-small-en-v1.5` (with its query instruction prefix) as a small dense baseline. Max sequence length 512. Raw `encode()` outputs are cast to float32 and L2-normalized before scoring.
- **Search:** exact (brute-force) search on GPU, no ANN index. BM25 via `bm25s` with English stopwords.
- **Quantization:** int8 uses one global scale fitted on the corpus (99.9th percentile of absolute values). Binary uses the sign of the mean-centered vector; ranking by Hamming distance is computed as a ±1 dot product. Rescoring takes the top 200 binary candidates and re-ranks them with float dot products.
- **Hybrid:** reciprocal rank fusion, c = 60, equal weights.
- **Contextual:** each document is split into fixed windows of 3 sentences. Chunks are ranked, then max-pooled to documents. The contextual model embeds the chunks of one document together (loaded via `AutoModel` with `trust_remote_code`). Queries are embedded either as single-chunk inputs to the contextual model or by the standard model.
- **Statistics:** nDCG@10 and recall@100; 2,000 bootstrap resamples over queries; paired differences for system comparisons.

## Limitations

- One run per configuration, one model size (0.6B), three BEIR datasets. Many comparisons were made without multiple-comparison correction, so treat borderline intervals cautiously.
- **Latency numbers are not meaningful for binary search.** The CPU benchmark uses a NumPy lookup-table popcount, which is slower than a BLAS float matmul (e.g. 23 ms vs 7.8 ms per query on FiQA). Optimized Hamming search (hardware popcount, SIMD) would behave differently. Only the quality and index-size results should be relied on.
- Chunking is fixed-size sentence windows; other chunk sizes and overlap were not tried.
- Relevance labels are document-level, which favors whole-document embedding over chunk-level retrieval.
- RRF uses equal weights. Weighted fusion or a reranker was not tested.
- `bge-small` is a small baseline, not a strong one; int8 and binary were compared on different models (int8 for bge-small, binary for pplx).

## Next steps

- Benchmarks built for contextual retrieval over long documents (e.g. ConTEB) and other chunk sizes.
- Weighted hybrid fusion and a reranker on top.
- A fast Hamming search implementation (e.g. in Rust) for realistic latency numbers.
- The 4B models.

## Repository layout

```
run_dense.py          # baselines, quantization study, hybrid
run_contextual.py     # contextual vs standard embeddings
retrieval/            # data loading, embedding + caching, search, metrics
results/              # generated tables and per-query scores
```

## References

- Perplexity `pplx-embed` models: https://huggingface.co/perplexity-ai
- BEIR benchmark: Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models* (2021)
- `bge-small-en-v1.5`: https://huggingface.co/BAAI/bge-small-en-v1.5

## Author

Omer Faruk Aydin — https://sites.google.com/view/omer-faruk-aydin/home