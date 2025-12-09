import os
import json

import numpy as np
from sentence_transformers import SentenceTransformer

EMB_DIR = "./embeddings"
EMB_MATRIX_PATH = os.path.join(EMB_DIR, "doc_embeddings.npy")
EMB_META_PATH = os.path.join(EMB_DIR, "doc_ids.json")

BM25_DOCS_PATH = "./bm25-files/docs00.json"
CASE_FILE = "./data/case_data.json"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class DenseRetriever:
    def __init__(
        self,
        emb_matrix_path=EMB_MATRIX_PATH,
        emb_meta_path=EMB_META_PATH,
        docs_path=BM25_DOCS_PATH,
        case_file=CASE_FILE,
        model_name=MODEL_NAME,
    ):
        if not os.path.exists(emb_matrix_path) or not os.path.exists(emb_meta_path):
            raise RuntimeError(
                "Embeddings not found. Run build_dense_embeddings.py first."
            )

        self.emb_matrix = np.load(emb_matrix_path)
        with open(emb_meta_path) as f:
            meta = json.load(f)
        self.doc_ids = meta["doc_ids"]

        self.docs_map = {}
        with open(docs_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                cid = str(entry["id"])
                self.docs_map[cid] = entry["contents"]

        with open(case_file) as f:
            data = json.load(f)["data"]
        self.cases = {str(c["id"]): c for c in data}

        self.model = SentenceTransformer(model_name)

    def search(self, query, top_k=10):
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores = np.dot(self.emb_matrix, q_emb)

        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            case_id = self.doc_ids[idx]
            score = float(scores[idx])
            text = self.docs_map.get(case_id, "")
            meta = self.cases.get(case_id, {})

            results.append({
                "rank": rank,
                "case_id": case_id,
                "score": score,
                "snippet": text[:300] + "...",
                "name": meta.get("name", ""),
                "court": meta.get("court", ""),
                "date": meta.get("date", ""),
                "cite": meta.get("cite", ""),
            })
        return results
