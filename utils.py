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
from multiprocessing import Pool, cpu_count
from PIL import Image

def encode_images_to_pil(img_dir):
    image_files = os.listdir(img_dir)
    image_files = sorted(image_files, key=lambda x: int(x.split("/")[-1].split('.')[0]))
    images = []
    
    for filename in image_files:
        images.append(Image.open(os.path.join(img_dir, filename)).convert("RGB"))
    
    return images

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
    """
    Extract and parse JSON from text using GPT to fix any formatting issues.
    
    Args:
        text: Input text that may contain JSON
        
    Returns:
        Parsed JSON as a dictionary
        
    Raises:
        ValueError: If JSON cannot be parsed
    """
    # Remove any markdown code block syntax first
    text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
    
    # Try to find and extract JSON-like substrings using a non-recursive pattern
    # Look for balanced curly braces with content between them
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_matches = re.finditer(json_pattern, text, re.DOTALL)
    
    for match in json_matches:
        try:
            json_str = match.group()
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue
            
    # If no valid JSON found, try parsing the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("No valid JSON found in text")

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
    # Filter for files that match the pattern of "1.png", "2.png", etc.
    image_files = [f for f in image_files if re.match(r'^\d+\.png$', f)]
    image_files = sorted(image_files, key=lambda x: int(x.split("/")[-1].split('.')[0]))
    image_urls = []
    
    for filename in image_files:
        img_path = os.path.join(img_dir, filename)
        image_url = encode_image_to_base64(img_path)
        if image_url:
            image_urls.append(image_url)
    
    return image_urls

def convert_pdf_to_png(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF file
    doc = fitz.open(input_file)

    # Iterate through each page of the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load the current page
        pix = page.get_pixmap()  # Render page to an image
        output_path = os.path.join(output_dir, f'{page_num + 1}.png')  # Define output path for the image
        pix.save(output_path)  # Save the image

    print(f"Conversion completed. Images are saved in '{output_dir}'.")

def pptx_to_pdf(input_file: str, output_dir: str) -> bool:
    """Convert a PPTX file to PDF using LibreOffice in Docker.
    
    Args:
        input_file: Path to the PPTX file
        output_dir: Directory to save the PDF file
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert paths to absolute paths
        input_file = os.path.abspath(input_file)
        input_dir = os.path.dirname(input_file)
        
        # Run LibreOffice conversion in Docker
        cmd = f'docker run --rm -v "{input_dir}:/data" libreoffice-converter libreoffice --headless --convert-to pdf --outdir /data "{os.path.basename(input_file)}"'
        
        # Execute command and check return code
        result = os.system(cmd)
        if result != 0:
            print(f"Error converting {input_file} to PDF")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error during PPTX to PDF conversion: {str(e)}")
        return False

def _parallel_wrapper(args):
    """
    Wrapper function for parallel processing that unpacks arguments.
    
    Args:
        args: Tuple of (func, arg) where:
            - func: The function to execute
            - arg: The argument to pass to the function
    """
    func, arg = args[0], args[1]
    try:
        return func(arg) if not isinstance(arg, tuple) else func(*arg)
    except Exception as e:
        print(f"Error processing with args {args}: {str(e)}")
        return None

def parallel_process(func, args_list, num_processes=None):
    """
    Run a function across multiple processes for parallel processing.
    
    Args:
        func: The function to run in parallel
        args_list: List of arguments, each containing the argument for one function call
        num_processes: Number of processes to use (defaults to CPU count)
        
    Returns:
        List of results from the function calls
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    # Prepare arguments for each item
    process_args = [(func, args) for args in args_list]
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(_parallel_wrapper, process_args),
            total=len(args_list),
            desc=f"Processing {len(args_list)} items with {num_processes} processes"
        ))
    
    return results