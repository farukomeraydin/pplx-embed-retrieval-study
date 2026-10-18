import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

_models = {}


def _announce(name, dev):
    print(f"Loading {name} on {dev}")
    if dev == "cpu":
        print("  WARNING: running on CPU (torch.cuda.is_available() is False). "
              "This will be very slow; install a CUDA build of PyTorch.")


def get_model(name):
    if name not in _models:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _announce(name, dev)
        _models[name] = SentenceTransformer(name, trust_remote_code=True, device=dev)
    return _models[name]


_ctx_models = {}


def get_ctx_model(name):
    """pplx-embed-context-v1 is NOT a SentenceTransformer model: the model card loads it with
    AutoModel(trust_remote_code=True) and calls .encode(list_of_docs_of_chunks)."""
    if name not in _ctx_models:
        from transformers import AutoModel

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _announce(name, dev)
        _ctx_models[name] = AutoModel.from_pretrained(name, trust_remote_code=True).to(dev).eval()
    return _ctx_models[name]


def free_model(name):
    _models.pop(name, None)
    _ctx_models.pop(name, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cache_path(name, tag, cache_dir):
    Path(cache_dir).mkdir(exist_ok=True)
    return Path(cache_dir) / f"{name.replace('/', '__')}__{tag}.npy"


def embed(name, texts, tag, prompt=None, batch_size=32, max_len=512, cache_dir="cache", block=8192):
    """Embed a list of strings -> float32 (n, d). Cached on disk by (model, tag).
    Encodes in blocks and checkpoints each block, so a crash never loses the whole run."""
    path = _cache_path(name, tag, cache_dir)
    if path.exists():
        return np.load(path)
    # Blank strings become zero-length sequences for models without special tokens
    # (pplx-embed) and crash the attention-mask code, so give them a placeholder.
    texts = [t if t and t.strip() else "N/A" for t in texts]
    m = get_model(name)
    try:
        m.max_seq_length = max_len
    except Exception:
        pass
    kw = {"prompt": prompt} if prompt else {}
    parts, part_paths = [], []
    for i in range(0, len(texts), block):
        pp = path.with_name(f"{path.stem}.part{i}.npy")
        part_paths.append(pp)
        if pp.exists():
            parts.append(np.load(pp))
            continue
        x = m.encode(texts[i : i + block], batch_size=batch_size, show_progress_bar=True, **kw)
        x = np.asarray(x)
        if i == 0:
            print(f"[{name}] raw output dtype={x.dtype}, shape={x.shape}")
        x = x.astype(np.float32)
        np.save(pp, x)
        parts.append(x)
        print(f"  embedded {min(i + block, len(texts))}/{len(texts)}")
    x = np.concatenate(parts)
    np.save(path, x)
    for pp in part_paths:
        pp.unlink(missing_ok=True)
    return x


def embed_context(name, chunked_docs, tag, docs_per_call=8, cache_dir="cache"):
    """Contextual model: input is a list of documents, each a list of chunk strings.
    Returns a flat float32 array (total_chunks, d), chunks in document order.
    NOTE: check the model card for the exact encode() contract; this follows the
    documented example where encode(list_of_docs) returns one (n_chunks, d) array per doc."""
    path = _cache_path(name, tag, cache_dir)
    if path.exists():
        return np.load(path)
    m = get_ctx_model(name)
    out = []
    for i in range(0, len(chunked_docs), docs_per_call):
        with torch.no_grad():
            res = m.encode(chunked_docs[i : i + docs_per_call])
        for doc_chunks, r in zip(chunked_docs[i : i + docs_per_call], res):
            r = np.asarray(r).astype(np.float32)
            assert r.shape[0] == len(doc_chunks), (r.shape, len(doc_chunks))
            out.append(r)
        if (i // docs_per_call) % 50 == 0:
            print(f"  context-embedded {i}/{len(chunked_docs)} docs")
    x = np.concatenate(out)
    np.save(path, x)
    return x
