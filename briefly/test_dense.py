from dense_retrieval import DenseRetriever

dr = DenseRetriever()
for r in dr.search("contract breach commercial lease", top_k=5):
    print(r["rank"], r["score"], r["name"], r["court"])