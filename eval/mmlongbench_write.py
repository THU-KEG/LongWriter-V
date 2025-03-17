import argparse
import os
from eval.utils import BaseEvaluator, print_res
from inference.api.gpt import GPT_Interface, DeepSeek_Interface, Gemini_Interface
from inference.api.claude import Claude_Interface
from utils import encode_image_to_base64
from datasets import load_dataset
from PIL import Image
import io
import pandas as pd
from tqdm import tqdm

class MMBenchWriteEvaluator(BaseEvaluator):
    """Evaluator for MMLongBench Write task"""
    def __init__(self, output_path, model_type='gpt-4o', method='vlm'):
        super().__init__(output_path) 
        self.model_type = model_type
        self.method = method
        self.cache_dir = "data/cache/MMLongBench-Write"
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_dataset()
        self._init_model()

    def _initialize_data(self):
        """Override parent's _initialize_data to load from HF dataset"""
        if os.path.exists(self.output_path):
            self.data = pd.read_excel(self.output_path)
        else:
            # Load dataset from Hugging Face
            dataset = load_dataset("THU-KEG/MMLongBench-Write", split="train")
            # Create DataFrame from dataset
            self.data = pd.DataFrame({
                'id': range(len(dataset)),
                'question': dataset['question'],
                'prediction': [None] * len(dataset)
            })
            
        # Initialize evaluation columns
        numeric_columns = ['length', 'length_score', 'relevance', 'accuracy', 'coherence', 
                          'clarity', 'breadth_depth', 'reading_experience', 'overall_quality_score']
        for col in numeric_columns:
            if col not in self.data.columns:
                self.data[col] = pd.Series(dtype='float64')
            else:
                self.data[col] = self.data[col].astype('float64')
        
        if 'quality_scores' not in self.data.columns:
            self.data['quality_scores'] = pd.Series(dtype='object')
        else:
            self.data['quality_scores'] = self.data['quality_scores'].astype('object')

    def _load_dataset(self):
        """Load dataset from Hugging Face"""
        self.dataset = load_dataset("THU-KEG/MMLongBench-Write", split="train")
        
        # Convert to DataFrame if output exists
        if os.path.exists(self.output_path):
            self.df = pd.read_excel(self.output_path)
        else:
            # Create DataFrame from dataset
            self.df = pd.DataFrame({
                'id': range(len(self.dataset)),
                'question': self.dataset['question'],
                'prediction': [None] * len(self.dataset)
            })

    def _init_model(self):
        """Initialize local models if needed"""
        self.model = None
        if self.model_type in ['glm-4-9b', 'qwen2.5-vl-7b', 'qwen2.5-vl-72b', 'minicpm-v2.6']:
            if self.model_type == 'glm-4-9b':
                from inference.local.glm import get_model as get_glm_model
                self.model = get_glm_model('9b-chat')
            elif self.model_type in ['qwen2.5-vl-7b', 'qwen2.5-vl-72b']:
                from inference.local.qwen2_5_vl import get_model as get_qwen2_5_vl_model
                size = '7b' if self.model_type == 'qwen2.5-vl-7b' else '72b'
                self.model = get_qwen2_5_vl_model(size)
            elif self.model_type == 'minicpm-v2.6':
                from inference.local.minicpm import get_model as get_minicpm_model
                self.model = get_minicpm_model('base')

    def _get_cached_image_path(self, idx):
        """Get or create cached image path"""
        cache_paths = []
        example = self.dataset[idx]
        
        for i, img in enumerate(example['images']):
            cache_path = os.path.join(self.cache_dir, f"{idx}_{i}.jpg")
            
            # Cache image if not already cached
            if not os.path.exists(cache_path):
                img.save(cache_path)
            
            cache_paths.append(cache_path)
            
        return cache_paths

    def _generate_caption(self, image_path):
        """Generate caption for images using GPT-4o"""
        prompt = open('eval/prompts/caption.txt', 'r').read()

        captions = []
        for p in image_path:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": encode_image_to_base64(p)}}]}]
            caption = GPT_Interface.call(model="gpt-4o", messages=messages)
            captions.append(caption)
        return captions

    def predict(self, idx, row):
        """Generate prediction using either VLM or Caption+LLM method"""
        image_paths = self._get_cached_image_path(idx)
        question = row['question']
        
        sample_params = {
            "temperature": 0.6,
            "max_tokens": 8192,
            "top_p": 0.95,
            "top_k": 0,
        }

        if self.method == 'caption_llm':
            # Generate captions first
            captions = self._generate_caption(image_paths)
            
            prompt = open('eval/prompts/llm_generate.txt', 'r').read()
            prompt = prompt.replace("$QUESTION$", question).replace("$CAPTIONS$", str(captions))
            
            messages = [{"role": "user", "content": prompt}]
        else:  # VLM method
            messages = [{"role": "user", "content": [
                {"type": "text", "text": question}
            ] + [{"type": "image", "image": "file://" + p} for p in image_paths]}]

        # Model-specific inference
        if self.model_type == 'gpt-4o':
            sample_params.pop('top_k')
            messages[0]["content"] = [
                {"type": "text", "text": question}
            ] + [{"type": "image_url", "image_url": {"url": encode_image_to_base64(p)}} for p in image_paths]
            res = GPT_Interface.call(model="gpt-4o-2024-08-06", messages=messages, use_cache=False, **sample_params)
        elif self.model_type == 'claude':
            sample_params['max_tokens'] = 4096
            messages = [{"role": "user", "content": 
                [{"type": "image", "source": {"type": "base64",
                                          "media_type": "image/jpeg" if p.endswith('.jpg') else "image/png",
                                          "data": encode_image_to_base64(p).split("base64,")[1]}} for p in image_paths]
                + [{"type": "text", "text": question}]}]
            res = Claude_Interface.call(model='claude-3-opus-20240229', messages=messages, **sample_params)
        elif self.model_type == 'gemini':
            sample_params.pop('top_k')
            messages[0]["content"] = [
                {"type": "text", "text": question}
            ] + [{"type": "image_url", "image_url": {"url": encode_image_to_base64(p)}} for p in image_paths]
            res = Gemini_Interface.call(model='gemini-1.5-pro', messages=messages, **sample_params)
        elif self.model_type in ['glm-4-9b', 'qwen2.5-vl-7b', 'qwen2.5-vl-72b']:
            sample_params["top_k"] = -1
            res = self.model.inference_vllm(messages, **sample_params)[0]
        elif self.model_type == 'minicpm-v2.6':
            messages = [{"role": "user", "content": 
                [Image.open(p).convert('RGB') for p in image_paths] +
                [question]}]
            res = self.model.inference(messages, **sample_params)
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")
            
        return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o', 
                      choices=['gpt-4o', 'claude', 'gemini', 'glm-4-9b', 
                              'qwen2.5-vl-7b', 'qwen2.5-vl-72b', 'minicpm-v2.6'],
                      help='Model type to use for prediction')
    parser.add_argument('--method', type=str, choices=['vlm', 'caption_llm'], 
                      default='vlm', help='Method to use for prediction')

    args = parser.parse_args()

    # Update output path based on method and model
    method_dir = 'caption_llm' if args.method == 'caption_llm' else 'vlm'
    output_dir = f'data/eval_res/MMLongBench_Write/{method_dir}'
    output_path = os.path.join(output_dir, f'{args.model}.xlsx')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    evaluator = MMBenchWriteEvaluator(
        output_path=output_path,
        model_type=args.model,
        method=args.method
    )
    evaluator.run()

    print_res({args.model: output_path})

if __name__ == "__main__":
    main() 