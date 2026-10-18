"""Minimal BEIR loader (no `beir` dependency)."""
import csv
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def load_beir(name, root="data", split="test"):
    root = Path(root)
    root.mkdir(exist_ok=True)
    d = root / name
    if not d.exists():
        z = root / f"{name}.zip"
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(URL.format(name), z)
        with zipfile.ZipFile(z) as f:
            f.extractall(root)

    corpus = {}
    with open(d / "corpus.jsonl", encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            corpus[r["_id"]] = (r.get("title", "") + "\n" + r["text"]).strip()

    queries = {}
    with open(d / "queries.jsonl", encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            queries[r["_id"]] = r["text"]

    qrels = {}
    with open(d / "qrels" / f"{split}.tsv", encoding="utf8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            qrels.setdefault(qid, {})[did] = int(score)

    qids = [q for q in queries if q in qrels]
    doc_ids = list(corpus)
    return {
        "name": name,
        "doc_ids": doc_ids,
        "docs": [corpus[i] if corpus[i].strip() else "N/A" for i in doc_ids],  # some corpora (e.g. fiqa) contain empty docs
        "qids": qids,
        "queries": [queries[q] if queries[q].strip() else "N/A" for q in qids],
        "qrels": qrels,
    }
