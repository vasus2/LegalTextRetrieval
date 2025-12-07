import os
import json
import random
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "./data/case_data_tokenized.json"
OUTPUT_DIR = "./bm25-files"
NUM_EXAMPLES = 5000  # adjust as needed
DEV_SPLIT = 0.1
TEST_SPLIT = 0.1
MAX_QUERY_WORDS = 100

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading tokenized case data...")
    with open(INPUT_FILE) as f:
        data = json.load(f)["data"]

    print(f"Loaded {len(data)} cases. Sampling {NUM_EXAMPLES} for BM25 prep...")
    data = random.sample(data, min(NUM_EXAMPLES, len(data)))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    docs_path = os.path.join(OUTPUT_DIR, "docs00.json")

    # ============================================================
    # 1️⃣ Write BM25 corpus (docs00.json)
    # ============================================================
    with open(docs_path, "w") as outfile:
        for i, case in enumerate(tqdm(data, desc="Writing docs")):
            doc = {
                "id": str(case["id"]),
                "contents": " ".join(case["text"]) if isinstance(case["text"], list) else str(case["text"])
            }
            json.dump(doc, outfile)
            outfile.write("\n")

    print(f"✅ Wrote BM25 corpus to {docs_path}")

    # ============================================================
    # 2️⃣ Split into train/dev/test
    # ============================================================
    random.shuffle(data)
    n_total = len(data)
    n_dev = int(n_total * DEV_SPLIT)
    n_test = int(n_total * TEST_SPLIT)

    dev_data = data[:n_dev]
    test_data = data[n_dev:n_dev + n_test]
    train_data = data[n_dev + n_test:]

    print(f"Split into: {len(train_data)} train / {len(dev_data)} dev / {len(test_data)} test")

    # ============================================================
    # 3️⃣ Write BM25 input queries
    # ============================================================
    def write_queries(filename, examples):
        with open(os.path.join(OUTPUT_DIR, filename), "w") as outfile:
            for idx, case in enumerate(examples):
                text = " ".join(case["text"]) if isinstance(case["text"], list) else str(case["text"])
                text = " ".join(text.split()[:MAX_QUERY_WORDS])
                outfile.write(f"{idx}\t{text}\n")

    write_queries("bm25_input_dev.tsv", dev_data)
    write_queries("bm25_input_test.tsv", test_data)

    print("✅ Wrote BM25 query files:")
    print(f"   {os.path.join(OUTPUT_DIR, 'bm25_input_dev.tsv')}")
    print(f"   {os.path.join(OUTPUT_DIR, 'bm25_input_test.tsv')}")

    # ============================================================
    # Done
    # ============================================================
    print("\n🎉 BM25 data preparation complete! You can now run Anserini commands such as:")
    print("   bin/IndexCollection -collection JsonCollection -input bm25-files -index indexes/bm25 -generator DefaultLuceneDocumentGenerator")