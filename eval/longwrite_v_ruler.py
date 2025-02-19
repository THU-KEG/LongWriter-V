import os
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from utils import count_words
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

class RulerEvaluator:
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path
        self.cache_dir = "data/cache/LongWrite-V-Ruler"
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_dataset()
        self._init_model()
        
    def _load_dataset(self):
        """Load dataset from Hugging Face"""
        self.dataset = load_dataset("THU-KEG/LongWrite-V-Ruler", split="train")
        
        # Convert to DataFrame if output exists
        if os.path.exists(self.output_path):
            self.df = pd.read_excel(self.output_path)
        else:
            # Create DataFrame from dataset
            self.df = pd.DataFrame({
                'id': range(len(self.dataset)),
                'question': self.dataset['question'],
                'L': self.dataset['L'],
                'prediction': [None] * len(self.dataset),
                'length': [None] * len(self.dataset)
            })
    
    def _init_model(self):
        """Initialize the model"""
        from inference.local.qwen2_vl import get_model
        self.model = get_model(self.model_path)
    
    def _get_cached_image_path(self, idx):
        """Get or create cached image path"""
        cache_paths = []
        example = self.dataset[idx]
        
        for i, img in enumerate(example['images']):
            cache_path = os.path.join(self.cache_dir, f"{idx}_{i}.jpg")
            
            # Cache image if not already cached
            if not os.path.exists(cache_path):
                # Convert image to PIL Image and save
                pil_image = Image.open(io.BytesIO(img['bytes']))
                pil_image.save(cache_path)
            
            cache_paths.append(cache_path)
            
        return cache_paths
    
    def evaluate(self):
        """Run evaluation"""
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            if not pd.isna(row['prediction']):
                continue
                
            question = row['question']
            image_paths = self._get_cached_image_path(idx)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        }
                    ] + [
                        {
                            "type": "image",
                            "image": "file://" + p
                        }
                        for p in image_paths
                    ]
                }
            ]
            
            sample_params = {
                "temperature": 0.9,
                "max_tokens": 8192,
                "top_p": 0.95,
                "top_k": -1,
            }
            
            res = self.model.inference_vllm(messages, **sample_params)[0]
            print(res)
            
            self.df.loc[idx, 'prediction'] = res
            self.df.loc[idx, 'length'] = count_words(res)
            
            # Save after each prediction
            self.df.to_excel(self.output_path, index=False)

def eval_ruler(model_path, output_path):
    evaluator = RulerEvaluator(model_path, output_path)
    evaluator.evaluate()
