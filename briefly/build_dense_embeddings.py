# build_dense_embeddings.py

import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

BM25_DOCS_PATH = "./bm25-files/docs00.json"
EMB_DIR = "./embeddings"
EMB_MATRIX_PATH = os.path.join(EMB_DIR, "doc_embeddings.npy")
EMB_META_PATH = os.path.join(EMB_DIR, "doc_ids.json")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # you can swap to a legal-specific model

def load_docs(docs_path: str):
    doc_ids = []
    docs = []
    with open(docs_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            cid = str(entry["id"])
            text = entry["contents"]
            doc_ids.append(cid)
            docs.append(text)
    return doc_ids, docs

def main():
    os.makedirs(EMB_DIR, exist_ok=True)

    print(f"Loading docs from {BM25_DOCS_PATH}...")
    doc_ids, docs = load_docs(BM25_DOCS_PATH)
    print(f"Loaded {len(docs)} documents.")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Encode in batches
    batch_size = 64
    all_embeddings = []

    for i in tqdm(range(0, len(docs), batch_size), desc="Encoding docs"):
        batch_texts = docs[i:i + batch_size]
        # sentence-transformers returns a numpy array by default
        batch_emb = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # important: makes cosine = dot product
        )
        all_embeddings.append(batch_emb)

    emb_matrix = np.vstack(all_embeddings)
    print("Embeddings shape:", emb_matrix.shape)

    np.save(EMB_MATRIX_PATH, emb_matrix)
    with open(EMB_META_PATH, "w") as f:
        json.dump({"doc_ids": doc_ids}, f)

    print(f"Saved embeddings to {EMB_MATRIX_PATH}")
    print(f"Saved doc id map to {EMB_META_PATH}")

if __name__ == "__main__":
    main()
