import json
import os
from dataclasses import dataclass
from rank_bm25 import BM25Okapi


@dataclass
class SearchResult:
    doc_id: str
    score: float
    snippet: str
    rank: int
    full_text: str = None
    case_name: str = None
    court: str = None
    date: str = None
    cite: str = None


class BM25Retriever:
    def __init__(self, corpus_path, metadata_path=None):
        self.corpus_path = corpus_path
        self.metadata_path = metadata_path
        
        self.doc_ids, self.docs, self.metadata = self._load_corpus()
        
        self.tokenized_corpus = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def _load_corpus(self):
        doc_ids = []
        docs = []
        
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        
        with open(self.corpus_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                doc_ids.append(entry["id"])
                docs.append(entry["contents"])
        
        metadata = {}
        if self.metadata_path and os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                metadata_json = json.load(f)
                cases = metadata_json.get("data", metadata_json) if isinstance(metadata_json, dict) else metadata_json
                
                for case in cases:
                    case_id = str(case.get("id", ""))
                    metadata[case_id] = {
                        "name": case.get("name", ""),
                        "court": case.get("court", ""),
                        "date": case.get("date", ""),
                        "cite": case.get("cite", "")
                    }
        
        return doc_ids, docs, metadata
    
    def search(self, query, top_k=10):
        if not query or not query.strip():
            return []
        
        tokenized_query = query.lower().split()
        
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = self.doc_ids[idx]
            doc_text = self.docs[idx]
            score = scores[idx]
            
            snippet = doc_text[:300]
            if len(doc_text) > 300:
                last_period = snippet.rfind('.')
                if last_period > 200:
                    snippet = snippet[:last_period + 1]
                else:
                    snippet += "..."
            
            meta = self.metadata.get(doc_id, {})
            
            result = SearchResult(
                doc_id=doc_id,
                score=float(score),
                snippet=snippet,
                rank=rank,
                full_text=doc_text,
                case_name=meta.get("name"),
                court=meta.get("court"),
                date=meta.get("date"),
                cite=meta.get("cite")
            )
            results.append(result)
        
        return results
    
    def get_corpus_size(self):
        return len(self.docs)
    
    def get_document_by_id(self, doc_id):
        try:
            idx = self.doc_ids.index(doc_id)
            return self.docs[idx]
        except ValueError:
            return None
