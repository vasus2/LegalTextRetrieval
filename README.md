# Briefly - Legal Precedent Retrieval System

This is a legal case search engine that allows you to find relevant case precedents using keywords (BM25).

## How to Run the App

### 1. Install Dependencies
Make sure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

### 2. Prepare Data & Build Indices
You need to run these scripts **once** to download data and build the search indices.

```bash
# 1. Download and process the legal dataset
python briefly/precedent_data_extraction.py

# 2. Build the BM25 search index
python briefly/bm25_pipeline.py

# 3. Build Dense/Hybrid search embeddings (takes a few minutes)
python briefly/build_dense_embeddings.py

# 4. Build the Citation Graph
python briefly/build_citation_graph.py
```

### 3. Run the Search App
Start the web interface:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Evaluation
To check the performance (MAP/Recall) of the system:
```bash
python evaluate_retrieval.py
```
