import argparse
import os
from eval.utils import BaseEvaluator, get_image_path, print_res
from inference.api.gpt import GPT_Interface, Gemini_Interface
from inference.api.claude import Claude_Interface
from PIL import Image
from utils import encode_image_to_base64
from inference.local.qwen2_vl import get_model as get_qwen2_vl_model
from inference.local.qwen2_5_vl import get_model as get_qwen2_5_vl_model
from inference.local.minicpm import get_model as get_minicpm_model

class VLMEvaluator(BaseEvaluator):
    """Evaluator that uses VLM models directly for prediction"""
    def __init__(self, data_path, output_path, model_type='qwen2-vl-7b'):
        super().__init__(data_path, output_path)
        self.model_type = model_type

    def predict(self, idx, row):
        """Generate prediction using VLM model"""
        image_path = get_image_path(idx, row)
        question = row['question']
        
        messages = [{"role": "user", "content": [
            {"type": "text", "text": question}
        ] + [{"type": "image", "image": "file://" + p} for p in image_path]}]

        sample_params = {
            "temperature": 0.6,
            "max_tokens": 8192,
            "top_p": 0.95,
            "top_k": 0,
        }
        
        res = None

        if self.model_type in ['gpt-4o', 'gemini']:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        }
                    ] +
                    [{
                        "type": "image_url",
                        "image_url": {"url": encode_image_to_base64(p)}} for p in image_path]
                }
            ]
        if self.model_type == 'gpt-4o':
            sample_params.pop('top_k')
            res = GPT_Interface.call(model='gpt-4o-2024-08-06', messages=messages, use_cache=False, **sample_params)
        elif self.model_type == 'claude':
            sample_params['max_tokens'] = 4096
            messages = [
                {
                    "role": "user",
                    "content":
                         [{"type": "image", "source": {"type": "base64",
                                                      "media_type": "image/jpeg" if os.path.splitext(p)[1].lower() == '.jpg' else "image/png",
                                                      "data": encode_image_to_base64(p).split("base64,")[1]}} for p in image_path]
                                                      + [
                            {
                                "type": "text",
                                "text": question
                            }
                        ]
                }
            ]
            res = Claude_Interface.call(model='claude-3-opus-20240229', messages=messages, **sample_params)
        elif self.model_type == 'gemini':
            sample_params.pop('top_k')
            res = Gemini_Interface.call(model='gemini-1.5-pro', messages=messages, **sample_params)
        elif self.model_type == 'longwriter-v-7b':
            if self.model is None:
                self.model = get_qwen2_5_vl_model('longwriter-v-7b')
            res = self.model.inference_vllm(messages, **sample_params)[0]
        elif self.model_type == 'minicpm-v2.6':
            if self.model is None:
                self.model = get_minicpm_model('base')
            messages = [{"role": "user", "content": 
                [Image.open(p).convert('RGB') for p in image_path] +
                [question]}]
            res = self.model.inference(messages, **sample_params)
        elif self.model_type == 'qwen2.5-vl-7b':
            sample_params["top_k"] = -1
            if self.model is None:
                self.model = get_qwen2_5_vl_model('7b')
            res = self.model.inference_vllm(messages, **sample_params)[0]
        elif self.model_type == 'qwen2.5-vl-72b':
            sample_params["top_k"] = -1
            if self.model is None:
                self.model = get_qwen2_5_vl_model('72b')
            res = self.model.inference_vllm(messages, **sample_params)[0]
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")
            
        return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2-vl-7b', help='Model type to use for prediction')
    parser.add_argument('--data_path', type=str, default='data/MMLongBench_Write.xlsx', help='Path to the data file')
    parser.add_argument('--output_path', type=str, default='data/eval_res/MMLongBench_Write/vlm/model.xlsx', help='Path to the output file')
    args = parser.parse_args()
    
    if args.output_path == 'data/eval_res/MMLongBench_Write/vlm/model.xlsx':
        args.output_path = args.output_path.replace('model.xlsx', f'{args.model}.xlsx')

    evaluator = VLMEvaluator(
        data_path=args.data_path,
        output_path=args.output_path,
        model_type=args.model
    )
    evaluator.run()

    print_res({args.model: args.output_path})

if __name__ == "__main__":
    main()
