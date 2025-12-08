# hybrid_retriever.py

import os
import json
from typing import List, Dict, Any

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
        docs_path: str = BM25_DOCS_PATH,
        case_file: str = CASE_FILE,
        emb_matrix_path: str = EMB_MATRIX_PATH,
        emb_meta_path: str = EMB_META_PATH,
        model_name: str = MODEL_NAME,
        alpha: float = 0.5,  # weight on BM25 vs. embeddings
    ):
        self.alpha = alpha

        # Load docs (id + text)
        self.doc_ids, self.docs = self._load_docs(docs_path)

        # BM25
        tokenized_corpus = [d.lower().split() for d in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Embeddings
        self.emb_matrix = np.load(emb_matrix_path)  # [N, d]
        with open(emb_meta_path) as f:
            meta = json.load(f)
        emb_doc_ids = meta["doc_ids"]

        # Sanity alignment (assumes same order; if not, we must reorder)
        if emb_doc_ids != self.doc_ids:
            raise RuntimeError(
                "Doc ID order mismatch between docs00.json and embeddings. "
                "You should build embeddings from the same docs00.json in the same order."
            )

        # Case metadata
        with open(case_file) as f:
            data = json.load(f)["data"]
        self.cases = {str(c["id"]): c for c in data}

        # Model for query encoding
        self.model = SentenceTransformer(model_name)

    def _load_docs(self, docs_path: str):
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

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_s = float(scores.min())
        max_s = float(scores.max())
        if max_s == min_s:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # BM25 scores
        q_tokens = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=float)

        # Dense scores
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        dense_scores = np.dot(self.emb_matrix, q_emb)

        # Normalize
        bm25_norm = self._normalize(bm25_scores)
        dense_norm = self._normalize(dense_scores)

        # Hybrid combination
        hybrid_scores = self.alpha * bm25_norm + (1.0 - self.alpha) * dense_norm

        top_indices = np.argsort(-hybrid_scores)[:top_k]

        results: List[Dict[str, Any]] = []
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
