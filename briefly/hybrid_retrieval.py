import os
import json

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BM25_DOCS_PATH = "./bm25-files/docs00.json"
CASE_FILE = "./data/case_data.json"
EMB_DIR = "./embeddings"
EMB_MATRIX_PATH = os.path.join(EMB_DIR, "doc_embeddings.npy")
EMB_META_PATH = os.path.join(EMB_DIR, "doc_ids.json")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class HybridRetriever:
    def __init__(
        self,
        docs_path=BM25_DOCS_PATH,
        case_file=CASE_FILE,
        emb_matrix_path=EMB_MATRIX_PATH,
        emb_meta_path=EMB_META_PATH,
        model_name=MODEL_NAME,
        alpha=0.5,
    ):
        self.alpha = alpha

        self.doc_ids, self.docs = self._load_docs(docs_path)

        tokenized_corpus = [d.lower().split() for d in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.emb_matrix = np.load(emb_matrix_path)
        with open(emb_meta_path) as f:
            meta = json.load(f)
        emb_doc_ids = meta["doc_ids"]

        if emb_doc_ids != self.doc_ids:
            raise RuntimeError(
                "Doc ID order mismatch between docs00.json and embeddings. "
                "You should build embeddings from the same docs00.json in the same order."
            )

        with open(case_file) as f:
            data = json.load(f)["data"]
        self.cases = {str(c["id"]): c for c in data}

        self.model = SentenceTransformer(model_name)

    def _load_docs(self, docs_path):
        doc_ids = []
        docs = []
        with open(docs_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                cid = str(entry["id"])
                text = entry.get("contents", "")
                doc_ids.append(cid)
                docs.append(text)
        return doc_ids, docs

    def _normalize(self, scores):
        if scores.size == 0:
            return scores
        min_s = float(scores.min())
        max_s = float(scores.max())
        if max_s == min_s:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def search(self, query, top_k=10):
        q_tokens = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=float)

        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        dense_scores = np.dot(self.emb_matrix, q_emb)

        bm25_norm = self._normalize(bm25_scores)
        dense_norm = self._normalize(dense_scores)

        hybrid_scores = self.alpha * bm25_norm + (1.0 - self.alpha) * dense_norm

        top_indices = np.argsort(-hybrid_scores)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            case_id = self.doc_ids[idx]
            text = self.docs[idx]
            meta = self.cases.get(case_id, {})
            results.append({
                "rank": rank,
                "case_id": case_id,
                "hybrid_score": float(hybrid_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "snippet": text[:300] + "...",
                "name": meta.get("name", ""),
                "court": meta.get("court", ""),
                "date": meta.get("date", ""),
                "cite": meta.get("cite", ""),
            })
        return results
