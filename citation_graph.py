"""
Citation Graph Module

Builds and queries a citation graph from legal case data.
Enables citation-based navigation and metrics calculation.
"""

import json
import os
from typing import Dict, List, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CitationGraph:
    """
    Citation graph for legal cases.
    
    Provides methods to:
    - Load pre-built citation graph
    - Query citing and cited cases
    - Calculate citation metrics
    """
    
    def __init__(self, graph_path: str = "./data/citation_graph.json"):
        """
        Load citation graph from disk.
        
        Args:
            graph_path: Path to citation graph JSON file
        """
        self.graph_path = graph_path
        self.nodes: Dict[str, dict] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)  # case_id -> [cited_case_ids]
        self.reverse_edges: Dict[str, List[str]] = defaultdict(list)  # case_id -> [citing_case_ids]
        
        if os.path.exists(graph_path):
            self._load_graph()
        else:
            logger.warning(f"Citation graph not found at {graph_path}. Run build_citation_graph.py first.")
    
    def _load_graph(self):
        """Load graph from JSON file."""
        try:
            with open(self.graph_path, 'r') as f:
                data = json.load(f)
            
            self.nodes = data.get('nodes', {})
            self.edges = defaultdict(list, data.get('edges', {}))
            self.reverse_edges = defaultdict(list, data.get('reverse_edges', {}))
            
            logger.info(f"Loaded citation graph: {len(self.nodes)} nodes, "
                       f"{sum(len(v) for v in self.edges.values())} edges")
        except Exception as e:
            logger.error(f"Error loading citation graph: {e}")
    
    def get_cited_cases(self, case_id: str) -> List[Dict[str, str]]:
        """
        Get all cases cited by the given case.
        
        Args:
            case_id: ID of the case
            
        Returns:
            List of dicts with case info (id, name, cite, citation_count)
        """
        case_id = str(case_id)
        cited_ids = self.edges.get(case_id, [])
        
        return [
            {
                'id': cid,
                'name': self.nodes.get(cid, {}).get('name', f'Case {cid}'),
                'cite': self.nodes.get(cid, {}).get('cite', ''),
                'citation_count': self.nodes.get(cid, {}).get('citation_count', 0)
            }
            for cid in cited_ids
            if cid in self.nodes
        ]
    
    def get_citing_cases(self, case_id: str) -> List[Dict[str, str]]:
        """
        Get all cases that cite the given case.
        
        Args:
            case_id: ID of the case
            
        Returns:
            List of dicts with case info (id, name, cite, citation_count)
        """
        case_id = str(case_id)
        citing_ids = self.reverse_edges.get(case_id, [])
        
        return [
            {
                'id': cid,
                'name': self.nodes.get(cid, {}).get('name', f'Case {cid}'),
                'cite': self.nodes.get(cid, {}).get('cite', ''),
                'citation_count': self.nodes.get(cid, {}).get('citation_count', 0)
            }
            for cid in citing_ids
            if cid in self.nodes
        ]
    
    def get_citation_metrics(self, case_id: str) -> Dict[str, int]:
        """
        Get citation metrics for a case.
        
        Args:
            case_id: ID of the case
            
        Returns:
            Dict with metrics: citation_count (cited by), cites_count (cites to)
        """
        case_id = str(case_id)
        
        return {
            'citation_count': len(self.reverse_edges.get(case_id, [])),
            'cites_count': len(self.edges.get(case_id, [])),
            'total_citations': self.nodes.get(case_id, {}).get('citation_count', 0)
        }
    
    def case_exists(self, case_id: str) -> bool:
        """Check if a case exists in the graph."""
        return str(case_id) in self.nodes
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get overall graph statistics.
        
        Returns:
            Dict with graph statistics
        """
        total_edges = sum(len(v) for v in self.edges.values())
        avg_citations = total_edges / len(self.nodes) if self.nodes else 0
        
        # Find most cited cases
        most_cited = sorted(
            [(cid, len(citing)) for cid, citing in self.reverse_edges.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': total_edges,
            'avg_citations_per_case': round(avg_citations, 2),
            'most_cited_cases': [
                {
                    'id': cid,
                    'name': self.nodes.get(cid, {}).get('name', ''),
                    'citation_count': count
                }
                for cid, count in most_cited
            ]
        }
