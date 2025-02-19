import argparse
from eval.utils import BaseEvaluator, get_image_path, print_res
from inference.api.gpt import GPT_Interface, DeepSeek_Interface
from utils import encode_image_to_base64

class CaptionLLMEvaluator(BaseEvaluator):
    """Evaluator that uses captioning and LLM for prediction"""
    def __init__(self, data_path, output_path, model_type='gpt-4o'):
        super().__init__(data_path, output_path)
        self.model_type = model_type
        
    def _generate_caption(self, image_path):
        """Generate caption for a single image"""
        prompt = """Please provide a detailed and comprehensive description of the image, paying special attention to both visual elements and textual content. Consider the following aspects:

1. Main Subject(s):
   - What are the primary objects, people, or figures in the image?
   - Their positioning, size, and prominence
   - Any diagrams, charts, or graphical elements

2. Textual Content:
   - All text visible in the image, including:
     * Headers, titles, or captions
     * Labels or annotations
     * Body text or paragraphs
     * Numbers, equations, or mathematical notation
   - The relationship between text and visual elements

3. Visual Details:
   - Colors, lighting, and overall composition
   - Textures and materials visible
   - Any notable patterns, designs, or visual hierarchies
   - Quality and clarity of text/figures

4. Information Structure:
   - How information is organized (e.g., flowcharts, tables, lists)
   - Connections or relationships indicated by arrows or lines
   - Legend or key elements if present
   - Reading order or flow of information

5. Technical Elements:
   - Presence of graphs, charts, or scientific figures
   - Any coordinate systems or axes
   - Units of measurement or scales
   - Technical symbols or notation

6. Context and Purpose:
   - The apparent purpose of the image (educational, technical, decorative, etc.)
   - Target audience or field of study
   - Any relevant domain-specific context

Please provide a clear, structured description that captures both the visual and textual elements, ensuring no significant details are omitted."""

        captions = []
        for p in image_path:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": encode_image_to_base64(p)}}]}]
            caption = GPT_Interface.call(model="gpt-4o", messages=messages)
            captions.append(caption)
        return captions
    
    def predict(self, idx, row):
        """Generate prediction using caption and LLM"""
        image_path = get_image_path(idx, row)
        captions = self._generate_caption(image_path)
        question = row['question']
        
        prompt = """
        Please analyze the following image captions and writing requirement carefully, then provide a detailed response that:
        1. Directly addresses the writing requirement
        2. Incorporates relevant details from the image captions
        3. Uses clear, well-structured writing
        4. Maintains appropriate tone and style for the context

Writing requirement:
$QUESTION$

Image captions:
$CAPTIONS$

Please provide a comprehensive response that fully satisfies the writing requirement while effectively utilizing the information from the image captions.
        """.replace("$QUESTION$", question).replace("$CAPTIONS$", str(captions))
        
        sample_params = {
            "temperature": 0.6,
            "max_tokens": 8192,
            "top_p": 0.95,
            "top_k": 0,
        }
        
        messages = [{"role": "user", "content": prompt}]        
        if self.model_type == 'gpt-4o':
            sample_params.pop('top_k')
            res = GPT_Interface.call(model="gpt-4o-2024-08-06", messages=messages, use_cache=False, **sample_params)
        elif self.model_type == 'deepseek-reasoner':
            sample_params.pop('top_k')
            res = DeepSeek_Interface.call(model='Pro/deepseek-ai/DeepSeek-R1', messages=messages, use_cache=False, **sample_params)
        elif self.model_type == 'glm-4-9b': # Done
            sample_params["top_k"] = -1
            if self.model is None:
                from inference.local.glm import get_model as get_glm_model
                self.model = get_glm_model('9b-chat')
            res = self.model.inference_vllm(messages, **sample_params)
        elif self.model_type == 'mistral-large-instruct-2407':
            sample_params["top_k"] = -1
            if self.model is None:
                from inference.local.mistral import get_model as get_mistral_model
                self.model = get_mistral_model('large-instruct-2407')
            res = self.model.inference(messages, **sample_params)
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")
            
        return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['gpt-4o', 'deepseek-reasoner', 'glm-4-9b', 'mistral-large-instruct-2407'], default='gpt-4o', help='Model type to use for prediction')
    parser.add_argument('--data_path', type=str, default='data/MMLongBench_Write.xlsx', help='Path to the data file')
    parser.add_argument('--output_path', type=str, default='data/eval_res/MMLongBench_Write/caption_llm/model.xlsx', help='Path to the output file')
    args = parser.parse_args()

    if args.output_path == 'data/eval_res/MMLongBench_Write/caption_llm/model.xlsx':
        args.output_path = args.output_path.replace('model.xlsx', f'{args.model}.xlsx')

    evaluator = CaptionLLMEvaluator(
        data_path=args.data_path,
        output_path=args.output_path,
        model_type=args.model
    )
    evaluator.run()

    print_res({args.model: args.output_path})

if __name__ == "__main__":
    main()