from utils.eval import calculate_metrics
from typing import List, Dict

def eval_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Calculate average BLEU and ROUGE scores across multiple script pairs.
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts to compare
        
    Returns:
        Dictionary containing average scores
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references must match number of hypotheses")
        
    scores = []
    for ref, hyp in zip(references, hypotheses):
        scores.append(calculate_metrics(ref, hyp))
    
    # Calculate averages
    avg_scores = {}
    for metric in scores[0].keys():
        avg_scores[metric] = sum(s[metric] for s in scores) / len(scores)
        
    return avg_scores
