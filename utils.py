import json
from tqdm import tqdm
import fitz
from pathlib import Path
import base64
import os
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from typing import List, Dict

def cal_text_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
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

def extract_json(text: str) -> Dict:
    # Try to extract JSON from markdown code blocks first
    import re
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    
    if json_match:
        # Found JSON in code blocks, parse the contents
        json_str = json_match.group(1)
    else:
        # No code blocks found, treat entire text as JSON
        json_str = text
        
    try:
        # Parse the JSON string
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}")

def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)
    
    total_count = chinese_char_count + english_word_count
    
    return total_count


def encode_image_to_base64(img_path):
    try:
        # Get file extension and corresponding MIME type
        extension = os.path.splitext(img_path)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(extension, 'image/jpeg')  # default to jpeg if unknown
        
        with open(img_path, 'rb') as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:{mime_type};base64,{base64_data}"
            
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        return None

def encode_images_to_base64(img_dir):
    image_files = os.listdir(img_dir)
    image_files = sorted(image_files, key=lambda x: int(x.split("/")[-1].split('.')[0]))
    image_urls = []
    
    for filename in image_files:
        img_path = os.path.join(img_dir, filename)
        image_url = encode_image_to_base64(img_path)
        if image_url:
            image_urls.append(image_url)
    
    return image_urls

def pdf_to_pngs(pdf_path: str, output_dir: str):
    """Convert a PDF file to PNG images.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the PNG images
    """
    # Create subdirectory based on PDF filename
    pdf_name = Path(pdf_path).stem
    pdf_output_dir = os.path.join(output_dir, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
        for page_num in tqdm(range(len(doc)), desc=f"Converting {Path(pdf_path).name}"):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            # Save to the PDF-specific subdirectory
            output_path = os.path.join(pdf_output_dir, f"{page_num + 1}.png")
            pix.save(output_path)
        doc.close()
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return False
    return True