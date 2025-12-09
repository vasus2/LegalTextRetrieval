"""
Build Citation Graph

Constructs a citation graph from case data by:
1. Loading all cases from case_data.json
2. Extracting citation strings from cites_to field
3. Fuzzy matching citations to case IDs in the corpus
4. Building bidirectional graph structure
5. Calculating citation metrics
6. Saving to citation_graph.json
"""

import json
import os
from typing import Optional
from collections import defaultdict
from rapidfuzz import fuzz, process
from tqdm import tqdm


def load_cases(case_file: str = "./data/case_data.json") -> dict:
    """Load case data from JSON file."""
    print(f"Loading cases from {case_file}...")
    with open(case_file, 'r') as f:
        data = json.load(f)
    
    cases = data['data']
    print(f"Loaded {len(cases)} cases")
    return cases


def build_citation_index(cases: list) -> dict:
    """
    Build an index mapping citation strings to case IDs.
    
    Args:
        cases: List of case dictionaries
        
    Returns:
        Dict mapping case_id to citation string
    """
    citation_index = {}
    
    for case in cases:
        case_id = str(case['id'])
        cite = case.get('cite', '')
        
        if cite:
            citation_index[case_id] = cite
    
    print(f"Built citation index with {len(citation_index)} entries")
    return citation_index


def match_citation_to_case(citation_str: str, citation_index: dict, threshold: int = 85) -> Optional[str]:
    """
    Match a citation string to a case ID using fuzzy matching.
    
    Args:
        citation_str: Citation string to match
        citation_index: Dict of case_id -> citation_string
        threshold: Minimum similarity score (0-100)
        
    Returns:
        Matched case_id or None
    """
    if not citation_str:
        return None
    
    # Try exact match first
    for case_id, cite in citation_index.items():
        if cite == citation_str:
            return case_id
    
    # Fuzzy match
    choices = list(citation_index.values())
    result = process.extractOne(
        citation_str,
        choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=threshold
    )
    
    if result:
        matched_cite, score, _ = result
        # Find case_id for matched citation
        for case_id, cite in citation_index.items():
            if cite == matched_cite:
                return case_id
    
    return None


def build_graph(cases: list) -> dict:
    """
    Build citation graph from cases.
    
    Returns:
        Dict with nodes, edges, and reverse_edges
    """
    print("\nBuilding citation graph...")
    
    # Build citation index for matching
    citation_index = build_citation_index(cases)
    
    # Initialize graph structures
    nodes = {}
    edges = defaultdict(list)  # case_id -> [cited_case_ids]
    reverse_edges = defaultdict(list)  # case_id -> [citing_case_ids]
    
    # Add all cases as nodes
    for case in cases:
        case_id = str(case['id'])
        nodes[case_id] = {
            'name': case.get('name', ''),
            'cite': case.get('cite', ''),
            'court': case.get('court', ''),
            'date': case.get('date', ''),
            'citation_count': 0  # Will be updated later
        }
    
    # Extract citation relationships
    matched_count = 0
    unmatched_count = 0
    
    for case in tqdm(cases, desc="Extracting citations"):
        case_id = str(case['id'])
        cites_to = case.get('cites_to', [])
        
        for citation_obj in cites_to:
            citation_str = citation_obj.get('cite', '')
            
            if not citation_str:
                continue
            
            # Try to match citation to a case in our corpus
            cited_case_id = match_citation_to_case(citation_str, citation_index)
            
            if cited_case_id:
                # Add edge: case_id cites cited_case_id
                if cited_case_id not in edges[case_id]:
                    edges[case_id].append(cited_case_id)
                
                # Add reverse edge: cited_case_id is cited by case_id
                if case_id not in reverse_edges[cited_case_id]:
                    reverse_edges[cited_case_id].append(case_id)
                
                matched_count += 1
            else:
                unmatched_count += 1
    
    # Update citation counts
    for case_id in nodes:
        nodes[case_id]['citation_count'] = len(reverse_edges.get(case_id, []))
    
    print(f"\nGraph construction complete:")
    print(f"  - Nodes: {len(nodes)}")
    print(f"  - Edges: {sum(len(v) for v in edges.values())}")
    print(f"  - Matched citations: {matched_count}")
    print(f"  - Unmatched citations: {unmatched_count}")
    print(f"  - Match rate: {matched_count / (matched_count + unmatched_count) * 100:.1f}%")
    
    return {
        'nodes': nodes,
        'edges': dict(edges),
        'reverse_edges': dict(reverse_edges)
    }


def save_graph(graph: dict, output_file: str = "./data/citation_graph.json"):
    """Save citation graph to JSON file."""
    print(f"\nSaving citation graph to {output_file}...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(graph, f, indent=2)
    
    print(f"Citation graph saved successfully!")


def print_stats(graph: dict):
    """Print graph statistics."""
    nodes = graph['nodes']
    edges = graph['edges']
    reverse_edges = graph['reverse_edges']
    
    total_edges = sum(len(v) for v in edges.values())
    avg_citations = total_edges / len(nodes) if nodes else 0
    
    # Find most cited cases
    most_cited = sorted(
        [(cid, len(citing)) for cid, citing in reverse_edges.items()],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    print("\n" + "="*60)
    print("CITATION GRAPH STATISTICS")
    print("="*60)
    print(f"Total cases: {len(nodes)}")
    print(f"Total citation edges: {total_edges}")
    print(f"Average citations per case: {avg_citations:.2f}")
    print(f"\nTop 5 most cited cases:")
    for i, (case_id, count) in enumerate(most_cited, 1):
        case_name = nodes[case_id]['name'][:60]
        print(f"  {i}. {case_name}")
        print(f"     Cited by {count} cases")
    print("="*60)


if __name__ == "__main__":
    # Load cases
    cases = load_cases()
    
    # Build graph
    graph = build_graph(cases)
    
    # Print statistics
    print_stats(graph)
    
    # Save graph
    save_graph(graph)
    
    print("\n✓ Citation graph build complete!")
    print("  Run the app to see citation features in action.")
