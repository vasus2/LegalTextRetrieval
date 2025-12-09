import json
import os
from typing import Optional
from collections import defaultdict
from rapidfuzz import fuzz, process
from tqdm import tqdm


def load_cases(case_file: str = "./data/case_data.json") -> dict:
    print(f"Loading cases from {case_file}...")
    with open(case_file, 'r') as f:
        data = json.load(f)
    
    cases = data['data']
    print(f"Loaded {len(cases)} cases")
    return cases


def build_citation_index(cases: list) -> dict:
    citation_index = {}
    
    for case in cases:
        case_id = str(case['id'])
        cite = case.get('cite', '')
        
        if cite:
            citation_index[case_id] = cite
    
    print(f"Built citation index with {len(citation_index)} entries")
    return citation_index


def match_citation_to_case(citation_str: str, citation_index: dict, threshold: int = 85) -> Optional[str]:
    if not citation_str:
        return None
    
    for case_id, cite in citation_index.items():
        if cite == citation_str:
            return case_id
    
    choices = list(citation_index.values())
    result = process.extractOne(
        citation_str,
        choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=threshold
    )
    
    if result:
        matched_cite, score, _ = result
        for case_id, cite in citation_index.items():
            if cite == matched_cite:
                return case_id
    
    return None


def build_graph(cases: list) -> dict:
    citation_index = build_citation_index(cases)
    
    nodes = {}
    edges = defaultdict(list)
    reverse_edges = defaultdict(list)
    
    for case in cases:
        case_id = str(case['id'])
        nodes[case_id] = {
            'name': case.get('name', ''),
            'cite': case.get('cite', ''),
            'court': case.get('court', ''),
            'date': case.get('date', ''),
            'citation_count': 0
        }
    
    matched_count = 0
    unmatched_count = 0
    
    for case in tqdm(cases, desc="Extracting citations"):
        case_id = str(case['id'])
        cites_to = case.get('cites_to', [])
        
        for citation_obj in cites_to:
            citation_str = citation_obj.get('cite', '')
            
            if not citation_str:
                continue
            
            cited_case_id = match_citation_to_case(citation_str, citation_index)
            
            if cited_case_id:
                if cited_case_id not in edges[case_id]:
                    edges[case_id].append(cited_case_id)
                
                if case_id not in reverse_edges[cited_case_id]:
                    reverse_edges[cited_case_id].append(case_id)
                
                matched_count += 1
            else:
                unmatched_count += 1
    
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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(graph, f, indent=2)
    
    print(f"Citation graph saved successfully!")


def print_stats(graph: dict):
    nodes = graph['nodes']
    edges = graph['edges']
    reverse_edges = graph['reverse_edges']
    
    total_edges = sum(len(v) for v in edges.values())
    avg_citations = total_edges / len(nodes) if nodes else 0
    
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
    cases = load_cases()
    graph = build_graph(cases)
    print_stats(graph)
    save_graph(graph)
    
    print("\n✓ Citation graph build complete!")
