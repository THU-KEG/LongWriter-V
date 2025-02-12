import argparse
from eval.utils import BaseEvaluator, get_image_path, print_task_res
from inference.api.gpt import GPT_Interface
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
        
        # Create messages with direct image input
        messages = [{"role": "user", "content": [
            {"type": "text", "text": question}
        ] + [{"type": "image", "image": "file://" + p} for p in image_path]}]

        sample_params = {
            "temperature": 0.6,
            "max_tokens": 8192,
            "top_p": 0.95,
            "top_k": 0,
        }
        
        if self.model_type in ['gpt-4o', 'gpt-4v']:
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
            res = GPT_Interface.call(model='gpt-4o-2024-05-13', messages=messages, use_cache=False, **sample_params)
        elif self.model_type == 'gpt-4v':
            sample_params.pop('top_k')
            res = GPT_Interface.call(model='gpt-4-vision-preview', messages=messages, use_cache=False, **sample_params)
        elif self.model_type == 'longwriter-v-7b-2.5':
            if self.model is None:
                self.model = get_qwen2_5_vl_model('longwriter-v-7b')
            res = self.model.inference_vllm(messages, **sample_params)[0]
        elif self.model_type == 'longwriter-v-7b-dpo-2.5':
            if self.model is None:
                self.model = get_qwen2_5_vl_model('longwriter-v-7b-dpo')
            res = self.model.inference_vllm(messages, **sample_params)[0]
        elif self.model_type == 'longwriter-v-7b-dpo-2.5-2':
            if self.model is None:
                self.model = get_qwen2_5_vl_model('longwriter-v-7b-dpo-2')
            res = self.model.inference_vllm(messages, **sample_params)[0]
        elif self.model_type == 'longwriter-v-7b-dpo-2.5-mixed-0':
            if self.model is None:
                self.model = get_qwen2_5_vl_model('longwriter-v-7b-dpo-mixed')
            res = self.model.inference_vllm(messages, **sample_params)[0]
        # elif self.model_type == 'qwen2-vl-7b':
        #     if self.model is None:
        #         self.model = get_qwen2_vl_model('7b', tensor_parallel_size=4)
        #     res = self.model.inference_vllm(messages, **sample_params)[0]
        # elif self.model_type == 'qwen2-vl-72b':
        #     if self.model is None:
        #         self.model = get_qwen2_vl_model('72b')
        #     res = self.model.inference_vllm(messages, **sample_params)[0]
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
        
        elif self.model_type == 'longwriter-v':
            if self.model is None:
                self.model = get_qwen2_vl_model('longwriter-v')
            res = self.model.inference(messages)
        elif self.model_type == 'longwriter-v-72b':
            if self.model is None:
                self.model = get_qwen2_vl_model('longwriter-v-72b')
            res = self.model.inference(messages)
        elif self.model_type == 'longwriter-v-7b-dpo':
            if self.model is None:
                self.model = get_qwen2_vl_model('longwriter-v-7b-dpo', tensor_parallel_size=4)
            res = self.model.inference_vllm(messages, **sample_params)[0]
        elif self.model_type == 'longwriter-v-7b-dpo-iter':
            if self.model is None:
                self.model = get_qwen2_vl_model('longwriter-v-7b-dpo-iter', tensor_parallel_size=4)
            res = self.model.inference_vllm(messages, **sample_params)[0]
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")
            
        return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2-vl-7b', help='Model type to use for prediction')
    parser.add_argument('--data_path', type=str, default='data/MMLongBench_Write.xlsx', help='Path to the data file')
    parser.add_argument('--output_path', type=str, default='data/eval_res/custom/vlm/model.xlsx', help='Path to the output file')
    args = parser.parse_args()
    
    if args.output_path == 'data/eval_res/custom/vlm/model.xlsx':
        args.output_path = args.output_path.replace('model.xlsx', f'{args.model}.xlsx')

#     \texttt{Claude-3-Opus-20240229} & 60.1 & 42.5 & 77.6 & 60.9 & 73.0 & 46.2 & 83.5 & 27.2 & 68.9 & 12.6 & 63.0 \\
#     \texttt{GPT-4o-2024-05-13} & 73.4 & 57.1 & 89.7 & 86.7 & \textbf{93.2} & 57.6 & 90.5 & 44.7 & \textbf{85.0} & 21.5 & 87.0 \\
#     \texttt{GPT-4-vision-preview} & 74.6 & 58.7 & \textbf{90.5} & 88.5 & 92.3 & 59.1 & \textbf{91.5} & 47.3 & 84.8 & 19.9 & \textbf{91.7} \\
# \texttt{MiniCPM-V2.6} & 40.9 & 20.4 & 61.4 & 24.9 & 49.6 & 24.2 & 69.9 & 8.2 & 49.2 & 11.5 & 50.5 \\
# \texttt{Qwen2.5-VL-7B-Instruct} & 51.1 & 39.0 & 63.3 & 63.3 & 56.8 & 37.9 & 68.9 & 35.7 & 52.7 & 5.2 & 58.8 \\
# \texttt{Qwen2.5-VL-72B-Instruct} & 63.7 & 49.7 & 77.7 & 79.3 & 75.7 & 48.1 & 81.2 & 36.0 & 69.5 & 32.7 & 75.5 \\
# \texttt{LongWriter-V-7B} & 82.0 & 85.4 & 78.6 & 68.4 & 77.0 & \textbf{89.1} & 80.8 & 88.0 & 73.1 & 86.3 & 77.8 \\
# \texttt{LongWriter-V-72B} & 82.7 & 81.6 & 83.7 & 67.3 & 86.4 & 85.4 & 84.8 & 78.3 & 76.9 & 90.8 & 86.6 \\
# \texttt{LongWriter-V-7B-DPO} & \textbf{83.3} & \textbf{85.8} & 80.7 & 67.6 & 75.9 & 88.8 & 84.5 & \textbf{89.2} & 75.4 & \textbf{92.6} & 75.0 \\
    evaluator = VLMEvaluator(
        data_path=args.data_path,
        output_path=args.output_path,
        model_type=args.model
    )
    evaluator.run()

    # print_res_head()
    print_task_res({args.model: args.output_path})

if __name__ == "__main__":
    main()
