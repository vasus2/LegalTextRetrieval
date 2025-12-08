"""
BM25 Retriever for legal case search.

This module provides a clean interface to the BM25 retrieval algorithm,
refactored from the original bm25_results.py script.
"""

import json
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi


@dataclass
class SearchResult:
    """
    Represents a single search result from BM25 retrieval.
    
    Attributes:
        doc_id: Unique identifier for the case document
        score: BM25 relevance score
        snippet: Short text excerpt from the document
        full_text: Complete document text (optional, for detailed view)
        rank: Position in the ranked results (1-indexed)
        case_name: Name/title of the case (if available)
        court: Court that issued the decision (if available)
        date: Date of the decision (if available)
        cite: Legal citation (if available)
    """
    doc_id: str
    score: float
    snippet: str
    rank: int
    full_text: Optional[str] = None
    case_name: Optional[str] = None
    court: Optional[str] = None
    date: Optional[str] = None
    cite: Optional[str] = None


class BM25Retriever:
    """
    BM25-based retrieval system for legal case documents.
    
    This class encapsulates the BM25 retrieval pipeline:
    - Loading the corpus from JSON lines format
    - Building the BM25 index
    - Running queries and returning ranked results
    
    Usage:
        retriever = BM25Retriever(corpus_path="bm25-files/docs00.json")
        results = retriever.search("contract breach damages", top_k=10)
    """
    
    def __init__(self, corpus_path: str, metadata_path: Optional[str] = None):
        """
        Initialize the BM25 retriever.
        
        Args:
            corpus_path: Path to the BM25 corpus file (JSON lines format)
            metadata_path: Optional path to case metadata JSON file
        """
        self.corpus_path = corpus_path
        self.metadata_path = metadata_path
        
        # Load corpus and metadata
        self.doc_ids, self.docs, self.metadata = self._load_corpus()
        
        # Build BM25 index
        self.tokenized_corpus = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def _load_corpus(self) -> Tuple[List[str], List[str], dict]:
        """
        Load the document corpus from JSON lines file.
        
        Returns:
            Tuple of (doc_ids, docs, metadata_dict)
            - doc_ids: List of document IDs
            - docs: List of document texts
            - metadata_dict: Dictionary mapping doc_id to metadata (name, court, etc.)
        """
        doc_ids = []
        docs = []
        
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        
        with open(self.corpus_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                doc_ids.append(entry["id"])
                docs.append(entry["contents"])
        
        # Load metadata if available
        metadata = {}
        if self.metadata_path and os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                metadata_json = json.load(f)
                # Handle both {"data": [...]} and [...] formats
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
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search the corpus using BM25 ranking.
        
        Args:
            query: Natural language search query
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects, ranked by relevance
        """
        if not query or not query.strip():
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = self.doc_ids[idx]
            doc_text = self.docs[idx]
            score = scores[idx]
            
            # Create snippet (first 300 chars or up to end of sentence)
            snippet = doc_text[:300]
            if len(doc_text) > 300:
                # Try to end at a sentence boundary
                last_period = snippet.rfind('.')
                if last_period > 200:  # Only if we have a reasonable amount of text
                    snippet = snippet[:last_period + 1]
                else:
                    snippet += "..."
            
            # Get metadata if available
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
    
    def get_corpus_size(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self.docs)
    
    def get_document_by_id(self, doc_id: str) -> Optional[str]:
        """
        Retrieve the full text of a document by its ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Full document text, or None if not found
        """
        try:
            idx = self.doc_ids.index(doc_id)
            return self.docs[idx]
        except ValueError:
            return None
