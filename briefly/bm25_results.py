from rank_bm25 import BM25Okapi
import pandas as pd
import json
import os

# load corpus
bm25_dir = os.path.join(os.path.dirname(__file__), "../bm25-files")
bm25_path = os.path.abspath(os.path.join(bm25_dir, "docs00.json"))

docs = []
with open(bm25_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        docs.append(entry["contents"])

print(f"Loaded {len(docs)} documents from {bm25_path}")

tokenized_corpus = [doc.lower().split() for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)
print("BM25 index built.\n")

# define test queries
queries = [
    "precedent ruling on due process",
    "freedom of speech first amendment cases",
    "contract breach and damages remedies",
    "tort negligence liability standard",
    "privacy rights under the fourth amendment"
]

top_k = 10
all_results = []

# run retrieval for each query
for qid, query in enumerate(queries, start=1):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_docs = [docs[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    # save top-k for this query
    for rank, (doc, score) in enumerate(zip(top_docs, top_scores), start=1):
        all_results.append({
            "Query ID": qid,
            "Query": query,
            "Rank": rank,
            "BM25 Score": round(score, 2),
            "Document Snippet": doc[:180] + "..."
        })

    print(f"Processed Query {qid}: {query}")

df_results = pd.DataFrame(all_results)
out_csv = os.path.join(bm25_dir, "bm25_multiquery_top10.csv")
df_results.to_csv(out_csv, index=False)

print(f"\n=== Sample of BM25 Retrieval Results ===")
print(df_results.head(15).to_string(index=False))
print(f"\nSaved all results to {out_csv}")