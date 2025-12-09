
import sys
import os
import json
import random
import logging
from tqdm import tqdm
from typing import List, Dict, Set

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'briefly'))

from retrieval.search_interface import SearchInterface
from retrieval.evaluator import Evaluator
from citation_graph import CitationGraph

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

CORPUS_PATH = "bm25-files/docs00.json"
METADATA_PATH = "data/case_data.json"
CITATION_GRAPH_PATH = "data/citation_graph.json"

def load_test_cases(limit: int = 100) -> List[Dict]:
    logger.info("Loading citation graph and finding test cases...")
    cg = CitationGraph(CITATION_GRAPH_PATH)
    
    with open(METADATA_PATH, 'r') as f:
        data = json.load(f)['data']
    
    candidates = []
    
    for case in data:
        case_id = str(case['id'])
        cited_cases = cg.get_cited_cases(case_id)
        
        if cited_cases and len(cited_cases) > 0:
            relevant_ids = {c['id'] for c in cited_cases}
            candidates.append({
                'id': case_id,
                'text': case.get('text', '')[:2000],
                'relevant_ids': relevant_ids
            })
    
    logger.info(f"Found {len(candidates)} candidates with valid ground truth citations.")
    if len(candidates) > limit:
        random.seed(42)
        test_samples = random.sample(candidates, limit)
    else:
        test_samples = candidates
        
    logger.info(f"Selected {len(test_samples)} test cases for evaluation.")
    return test_samples

def run_evaluation(test_cases: List[Dict], methods: List[str], top_k: int = 20):
    logger.info(f"Initializing Search Interface...")
    search_interface = SearchInterface(CORPUS_PATH, METADATA_PATH)
    evaluator = Evaluator()
    
    final_results = {}
    
    for method in methods:
        logger.info(f"\nEvaluate Method: {method.upper()}")
        ap_scores = []
        recall_scores = []
        
        for case in tqdm(test_cases, desc=f"evaluating {method}"):
            query = case['text']
            if isinstance(query, list):
                query = " ".join(query)
            relevant_ids = case['relevant_ids']
            
            try:
                results = search_interface.search(query, top_k=top_k, method=method)
                retrieved_ids = [str(r['case_id']) for r in results]
                
                ap = evaluator.calculate_ap(retrieved_ids, relevant_ids)
                ap_scores.append(ap)
                
                recall = evaluator.calculate_recall_at_k(retrieved_ids, relevant_ids, top_k)
                recall_scores.append(recall)
                
            except Exception as e:
                logger.error(f"Error searching case {case['id']}: {e}")
                
        mean_ap = evaluator.calculate_map(ap_scores)
        mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        
        final_results[method] = {
            "MAP": mean_ap,
            f"Recall@{top_k}": mean_recall
        }
        
        logger.info(f"Result for {method}: MAP = {mean_ap:.4f} | Recall@{top_k} = {mean_recall:.4f}")

    return final_results

if __name__ == "__main__":
    if not os.path.exists(CITATION_GRAPH_PATH):
        logger.error("Citation graph not found. Please run 'python briefly/build_citation_graph.py' first.")
        sys.exit(1)
    test_cases = load_test_cases(limit=100)
    
    if not test_cases:
        logger.error("No test cases found. Ensure your dataset has citations that link to other cases in the corpus.")
        sys.exit(1)
    TARGET_METHODS = ["bm25"]
    
    try:
        import dense_retrieval
        TARGET_METHODS.append("dense")
    except ImportError:
        logger.warning("Dense retrieval not available/installed.")

    try:
        import hybrid_retrieval
        TARGET_METHODS.append("hybrid")
    except ImportError:
        logger.warning("Hybrid retrieval not available/installed.")

    results = run_evaluation(test_cases, TARGET_METHODS, top_k=20)
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY (Ground Truth: Citation Graph)")
    print("="*50)
    for method, scores in results.items():
        print(f"{method.upper():<10} | MAP: {scores['MAP']:.4f} | Recall@20: {scores[f'Recall@20']:.4f}")
    print("="*50)
