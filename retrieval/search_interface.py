from typing import List, Dict, Optional
from retrieval.bm25_retriever import BM25Retriever, SearchResult


class SearchInterface:
    def __init__(self, corpus_path: str, metadata_path: Optional[str] = None):
        self.corpus_path = corpus_path
        self.metadata_path = metadata_path
        
        self.bm25_retriever = BM25Retriever(
            corpus_path=corpus_path,
            metadata_path=metadata_path
        )
        
        self.current_method = "bm25"
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        method: str = "bm25"
    ) -> List[Dict]:
        if method == "bm25":
            return self._search_bm25(query, top_k)
        elif method == "embedding":
            raise NotImplementedError("Embedding-based search not yet implemented")
        elif method == "hybrid":
            raise NotImplementedError("Hybrid search not yet implemented")
        else:
            raise ValueError(f"Unknown search method: {method}")
    
    def _search_bm25(self, query: str, top_k: int) -> List[Dict]:
        if method == "bm25":
            return self._search_bm25(query, top_k)
        elif method == "embedding":
            raise NotImplementedError("Embedding-based search not yet implemented")
        elif method == "hybrid":
            raise NotImplementedError("Hybrid search not yet implemented")
        else:
            raise ValueError(f"Unknown search method: {method}")
    
    def _search_bm25(self, query: str, top_k: int) -> List[Dict]:
        results = self.bm25_retriever.search(query, top_k=top_k)
        
        return [
            {
                'rank': r.rank,
                'case_id': r.doc_id,
                'title': r.case_name or f"Case {r.doc_id}",
                'citation': r.cite,
                'court': r.court,
                'date': r.date,
                'score': r.score,
                'snippet': r.snippet,
                'full_text': r.full_text
            }
            for r in results
        ]
    
    def get_corpus_size(self) -> int:
        return self.bm25_retriever.get_corpus_size()
    
    def get_available_methods(self) -> List[str]:
        return ["bm25"]
    
    def set_method(self, method: str):
        if method not in self.get_available_methods():
            raise ValueError(f"Method {method} not available")
        self.current_method = method
