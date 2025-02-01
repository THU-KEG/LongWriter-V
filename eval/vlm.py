import argparse
from eval.utils import BaseEvaluator, get_image_path, print_res, print_res_head
from inference.local.qwen2_vl import get_model as get_qwen2_vl_model
from inference.local.qwen2_5_vl import get_model as get_qwen2_5_vl_model

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
        
        if self.model_type == 'qwen2-vl-7b':
            if self.model is None:
                self.model = get_qwen2_vl_model('7b')
            res = self.model.inference(messages)
        elif self.model_type == 'qwen2-vl-72b':
            if self.model is None:
                self.model = get_qwen2_vl_model('72b')
            res = self.model.inference(messages)
        elif self.model_type == 'qwen2.5-vl-7b':
            if self.model is None:
                self.model = get_qwen2_5_vl_model('7b')
            res = self.model.inference(messages)
        elif self.model_type == 'qwen2.5-vl-72b':
            if self.model is None:
                self.model = get_qwen2_5_vl_model('72b')
            res = self.model.inference(messages)
        elif self.model_type == 'longwriter-v':
            if self.model is None:
                self.model = get_qwen2_vl_model('longwriter-v')
            res = self.model.inference(messages)
        elif self.model_type == 'longwriter-v-72b':
            if self.model is None:
                self.model = get_qwen2_vl_model('longwriter-v-72b')
            res = self.model.inference(messages)
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")
            
        return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['qwen2-vl-7b', 'qwen2-vl-72b', 'qwen2.5-vl-7b', 'qwen2.5-vl-72b', 'longwriter-v', 'longwriter-v-72b'], default='qwen2-vl-7b', help='Model type to use for prediction')
    parser.add_argument('--data_path', type=str, default='data/MMLongBench_Write.xlsx', help='Path to the data file')
    parser.add_argument('--output_path', type=str, default='data/MMLongBench_Write_VLM.xlsx', help='Path to the output file')
    args = parser.parse_args()

    evaluator = VLMEvaluator(
        data_path=args.data_path,
        output_path=args.output_path,
        model_type=args.model
    )
    evaluator.run()

    print_res_head()
    print_res({args.model: args.output_path})

if __name__ == "__main__":
    main()
