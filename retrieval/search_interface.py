from typing import List, Dict, Optional
from retrieval.bm25_retriever import BM25Retriever, SearchResult


import sys
import os
import logging

try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'briefly'))
    from dense_retrieval import DenseRetriever
    from hybrid_retrieval import HybridRetriever
    DENSE_AVAILABLE = True
except ImportError:
    DENSE_AVAILABLE = False


class SearchInterface:
    def __init__(self, corpus_path: str, metadata_path: Optional[str] = None):
        self.corpus_path = corpus_path
        self.metadata_path = metadata_path
        
        self.bm25_retriever = BM25Retriever(
            corpus_path=corpus_path,
            metadata_path=metadata_path
        )

        self.dense_retriever = None
        self.hybrid_retriever = None

        if DENSE_AVAILABLE:
            try:
                self.dense_retriever = DenseRetriever()
                self.hybrid_retriever = HybridRetriever(alpha=0.6)
            except Exception as e:
                logging.warning(f"Failed to initialize dense/hybrid retrievers: {e}")
        
        self.current_method = "bm25"
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        method: str = "bm25"
    ) -> List[Dict]:
        if method == "bm25":
            return self._search_bm25(query, top_k)
        elif method == "dense" or method == "embedding":
            return self._search_dense(query, top_k)
        elif method == "hybrid":
            return self._search_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown search method: {method}")
            
    def _search_dense(self, query: str, top_k: int) -> List[Dict]:
        if not self.dense_retriever:
            raise NotImplementedError("Dense retrieval is not available (check logs for init errors)")
            
        results = self.dense_retriever.search(query, top_k)
        return self._standardize_results(results, "Semantic")

    def _search_hybrid(self, query: str, top_k: int) -> List[Dict]:
        if not self.hybrid_retriever:
            raise NotImplementedError("Hybrid retrieval is not available")
            
        results = self.hybrid_retriever.search(query, top_k)
        return self._standardize_results(results, "Hybrid")
    
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
                'score_type': 'BM25',
                'snippet': r.snippet,
                'full_text': r.full_text
            }
            for r in results
        ]

    def _standardize_results(self, results: List[Dict], score_type: str) -> List[Dict]:
        """Convert varied result formats to standard UI format"""
        standardized = []
        for r in results:
            standardized.append({
                'rank': r['rank'],
                'case_id': r['case_id'],
                'title': r.get('name', f"Case {r['case_id']}"),
                'citation': r.get('cite', None),
                'court': r.get('court', None),
                'date': r.get('date', None),
                'score': r.get('score', r.get('hybrid_score', 0)),
                'score_type': score_type,
                'snippet': r.get('snippet', ''),
                'full_text': r.get('snippet', '') # dense/hybrid don't return full text yet
            })
        return standardized
    
    def get_corpus_size(self) -> int:
        return self.bm25_retriever.get_corpus_size()
    
    def get_available_methods(self) -> List[str]:
        methods = ["bm25"]
        if self.dense_retriever:
            methods.append("dense")
        if self.hybrid_retriever:
            methods.append("hybrid")
        return methods
    
    def set_method(self, method: str):
        if method not in self.get_available_methods():
            raise ValueError(f"Method {method} not available")
        self.current_method = method
