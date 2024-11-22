from utils.eval import calculate_metrics
import os
from pathlib import Path
from PIL import Image
from typing import List, Dict
import matplotlib.pyplot as plt
from inference.local import ModelManager
from tqdm import tqdm
import json

model_manager = ModelManager()

def eval_metrics(references: List[str], hypotheses: List[str], output_path: str = 'outputs/eval/metrics') -> Dict[str, float]:
    os.makedirs(output_path, exist_ok=True)
    """
    Calculate average BLEU and ROUGE scores across multiple script pairs.
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts to compare
        output_path: Path where the visualization and JSON should be saved (default: 'outputs/eval/metrics')
        
    Returns:
        Dictionary containing average scores
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references must match number of hypotheses")
        
    scores = []
    for ref, hyp in tqdm(zip(references, hypotheses), total=len(references), desc="Calculating metrics"):
        scores.append(calculate_metrics(ref, hyp))
    
    # Calculate averages
    avg_scores = {}
    for metric in scores[0].keys():
        avg_scores[metric] = sum(s[metric] for s in scores) / len(scores)
    
    # Add visualization of individual scores
    fig, axes = plt.subplots(len(scores[0]), 1, figsize=(10, 4*len(scores[0])))
    if len(scores[0]) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(scores[0].keys()):
        metric_scores = [s[metric] for s in scores]
        axes[idx].plot(range(len(metric_scores)), metric_scores, marker='o', linestyle='-')
        axes[idx].set_title(f'{metric} Scores by Script')
        axes[idx].set_xlabel('Script Index')
        axes[idx].set_ylabel(metric)
        axes[idx].axhline(y=avg_scores[metric], color='r', linestyle='--', label='Average')
        axes[idx].grid(True)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'scores_by_script.png'))
    plt.close()
        
    # Add JSON output
    json_path = os.path.join(output_path, 'metrics_scores.json')
    with open(json_path, 'w') as f:
        json.dump(avg_scores, f, indent=4)

    print(f'Results saved to: {json_path}')
        
    return avg_scores

def eval_rm(imgs: List[Image.Image], scripts: List[str], output_path: str = 'outputs/eval/rm') -> Dict[str, float]:
    os.makedirs(output_path, exist_ok=True)
    """
    Evaluate lecture script and image pairs using reward model and visualize scores.
    
    Args:
        imgs: List of lecture slide images
        scripts: List of lecture transcripts to evaluate
        output_path: Path to save visualization plot and JSON
        
    Returns:
        Dictionary containing average scores for each metric
    """
    if len(imgs) != len(scripts):
        raise ValueError("Number of images must match number of scripts")
        
    scores = []
    for img, script in tqdm(zip(imgs, scripts), total=len(imgs), desc="Evaluating with reward model"):
        prompt = f"""Instructions:

You are provided with a segment of a lecture slide and its corresponding transcript. Evaluate the transcript based on the following criteria:

1. Faithfulness to the Slide: How accurately does the transcript represent the information on the provided PowerPoint slide?
2. Clarity of Language: How clear and understandable is the language used in the transcript?
3. Structure and Organization: How well is the transcript structured and organized?
4. Inspirational Value: How inspiring and engaging is the transcript?

Rate each criterion on a scale from 1 to 5, where 1 is the lowest and 5 is the highest. Provide your ratings in the format: Faithfulness, Clarity, Structure, Inspirational (e.g., 3, 4, 2, 5).

Transcript:
{script}

Output:
"""
        msgs = [{'role': 'user', 'content': [img, prompt]}]
        response = model_manager.inference_rm(msgs)
        scores.append([float(x) for x in response.split(', ')])

    # Calculate averages
    metrics = ['Faithfulness', 'Clarity', 'Structure', 'Inspirational'] 
    avg_scores = {}
    for i, metric in enumerate(metrics):
        avg_scores[metric] = sum(score[i] for score in scores) / len(scores)

    # Visualize scores
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        metric_scores = [score[idx] for score in scores]
        axes[idx].plot(range(len(metric_scores)), metric_scores, marker='o', linestyle='-')
        axes[idx].set_title(f'{metric} Scores by Script')
        axes[idx].set_xlabel('Script Index')
        axes[idx].set_ylabel('Score')
        axes[idx].axhline(y=avg_scores[metric], color='r', linestyle='--', label='Average')
        axes[idx].grid(True)
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'scores_by_script.png'))
    plt.close()

    # Add JSON output
    json_path = os.path.join(output_path, 'rm_scores.json')
    with open(json_path, 'w') as f:
        json.dump({
            'average_scores': avg_scores,
            'individual_scores': [
                {metric: score[i] for i, metric in enumerate(metrics)}
                for score in scores
            ]
        }, f, indent=4)

    print(f'Results saved to: {json_path}')

    return avg_scores

def eval_reranker(references: List[str], hypotheses: List[str], output_path: str = 'outputs/eval/reranker') -> Dict[str, float]:
    os.makedirs(output_path, exist_ok=True)
    pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    scores = model_manager.inference_reranker(pairs)
    avg_score = sum(scores) / len(scores)
    scores_json = {
        'average_score': avg_score,
        'individual_scores': scores
    }
    with open(os.path.join(output_path, 'reranker_scores.json'), 'w') as f:
        json.dump(scores_json, f, indent=4)
    return avg_score


if __name__ == '__main__':
    img_root = Path(__file__).parent.parent / 'data/bio2/pngs'
    ground_truth_root = Path(__file__).parent.parent / 'data/bio2/scripts_online'
    old_script_root = Path(__file__).parent.parent / 'data/bio2/scripts'
    script_root = Path(__file__).parent.parent / 'data/bio2/results_3'

    from lecgen.lecgen import get_scripts, encode_images_to_pil
    ground_truth_scripts = get_scripts(ground_truth_root)
    results_3_scripts = get_scripts(script_root)
    old_scripts = get_scripts(old_script_root)
    imgs = encode_images_to_pil(img_root)
    metrics_score = eval_metrics(ground_truth_scripts, results_3_scripts, output_path='outputs/eval/metrics/results_3')
    metrics_score = eval_metrics(ground_truth_scripts, old_scripts, output_path='outputs/eval/metrics/old')
    rm_score = eval_rm(imgs, results_3_scripts, output_path='outputs/eval/rm/results_3')
    rm_score = eval_rm(imgs, old_scripts, output_path='outputs/eval/rm/old')
    rm_score = eval_rm(imgs, ground_truth_scripts, output_path='outputs/eval/rm/ground_truth')
    reranker_score = eval_reranker(ground_truth_scripts, old_scripts, output_path='outputs/eval/reranker/old')
    reranker_score = eval_reranker(ground_truth_scripts, results_3_scripts, output_path='outputs/eval/reranker/results_3')
    print(reranker_score)
