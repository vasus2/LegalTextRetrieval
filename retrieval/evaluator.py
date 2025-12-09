from typing import List, Dict, Set

class Evaluator:
    @staticmethod
    def calculate_ap(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        if not relevant_ids:
            return 0.0
            
        score = 0.0
        num_hits = 0
        
        for k, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                num_hits += 1
                precision_at_k = num_hits / k
                score += precision_at_k
                
        return score / len(relevant_ids)

    @staticmethod
    def calculate_map(ap_scores: List[float]) -> float:
        if not ap_scores:
            return 0.0
        return sum(ap_scores) / len(ap_scores)

    @staticmethod
    def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
            
        top_k = set(retrieved_ids[:k])
        intersection = top_k.intersection(relevant_ids)
        return len(intersection) / len(relevant_ids)

    @staticmethod
    def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
            
        top_k = set(retrieved_ids[:k])
        intersection = top_k.intersection(relevant_ids)
        return len(intersection) / len(relevant_ids)

    @staticmethod
    def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
            
        top_k = set(retrieved_ids[:k])
        intersection = top_k.intersection(relevant_ids)
        return len(intersection) / len(relevant_ids)
        
    @staticmethod
    def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
            
        top_k = set(retrieved_ids[:k])
        intersection = top_k.intersection(relevant_ids)
        return len(intersection) / len(relevant_ids)
