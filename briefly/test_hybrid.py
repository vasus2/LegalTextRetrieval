# test_hybrid.py
from hybrid_retrieval import HybridRetriever

hr = HybridRetriever(alpha=0.6)  # 0.6 BM25, 0.4 dense; you can tune
query = "contract breach commercial lease"

for r in hr.search(query, top_k=5):
    print(
        r["rank"],
        f"hybrid={r['hybrid_score']:.3f}",
        f"bm25={r['bm25_score']:.3f}",
        f"dense={r['dense_score']:.3f}",
        r["name"],
        r["court"],
    )
