from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from typing import List, Dict

def calculate_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate BLEU and ROUGE scores between reference and hypothesis texts.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text to compare against reference
        
    Returns:
        Dictionary containing BLEU and ROUGE scores
    """
    # Prepare texts for BLEU scoring
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    # Calculate BLEU score with smoothing
    smooth = SmoothingFunction()
    bleu_score = sentence_bleu(
        [ref_tokens], 
        hyp_tokens,
        smoothing_function=smooth.method1
    )
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)
    
    return {
        'bleu': bleu_score,
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure, 
        'rougeL': rouge_scores['rougeL'].fmeasure
    }

