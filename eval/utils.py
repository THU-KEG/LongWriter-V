import os
import pandas as pd
from utils import count_words, extract_json, encode_image_to_base64
from inference.api.gpt import GPT_Interface
from tqdm import tqdm
from abc import ABC, abstractmethod

dims = ['Relevance', 'Accuracy', 'Coherence', 'Clarity', 'Breadth and Depth', 'Reading Experience']

def get_image_path(idx, row):
    """Get image path(s) from row data"""
    image_base_path = os.getenv('IMAGE_BASE_PATH', 'data/MMLongBench_Write')
    image_path = row['image_path']
    if isinstance(image_path, str) and image_path.startswith('[') and image_path.endswith(']'):
        image_paths = eval(image_path)
        return [os.path.join(image_base_path, p) for p in image_paths]
    else:
        return [os.path.join(image_base_path, str(idx) + '.jpg')]

class BaseEvaluator(ABC):
    """Base class for all evaluators"""
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.model = None
        self.data = None
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize the data DataFrame with required columns"""
        if os.path.exists(self.output_path):
            self.data = pd.read_excel(self.output_path)
        else:
            self.data = pd.read_excel(self.data_path)
        
        # Initialize columns
        if 'prediction' not in self.data.columns:
            self.data['prediction'] = pd.Series(dtype='object')
        else:
            self.data['prediction'] = self.data['prediction'].astype('object')
            
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
    
    @abstractmethod
    def predict(self, idx, row):
        """Generate prediction for a single row"""
        pass
    
    def evaluate_quality(self, idx, row):
        """Evaluate quality of prediction using VLM"""
        task_description = """You are an expert in evaluating text quality. Please evaluate the quality of an AI assistant's response to a user's writing request with several corresponding images. Be as strict as possible.

You need to evaluate across the following six dimensions, with scores ranging from 1 to 5. The scoring criteria from 5 to 1 for each dimension are as follows:

1. Relevance: From content highly relevant and fully applicable to the user's request and images to completely irrelevant or inapplicable.

2. Accuracy: From content completely accurate with no factual errors or misleading information to content with numerous errors and highly misleading.

3. Coherence: From clear structure with smooth logical connections to disorganized structure with no coherence.

4. Clarity: From clear language, rich in detail, and easy to understand to confusing expression with minimal details.

5. Breadth and Depth: From both broad and deep content with a lot of information to seriously lacking breadth and depth with minimal information.

6. Reading Experience: From excellent reading experience, engaging and easy to understand content to very poor reading experience, boring and hard to understand content.

Please evaluate the quality of the following response to a user's request according to the above requirements.

<User Request>

$INST$

</User Request>

<Response>

$RESPONSE$

</Response>

Please evaluate the quality of the response. You must first provide a brief analysis of its quality, then give a comprehensive analysis with scores for each dimension. The output must strictly follow the JSON format: {"Analysis": ..., "Relevance": ..., "Accuracy": ..., "Coherence": ..., "Clarity": ..., "Breadth and Depth": ..., "Reading Experience": ...}. You do not need to consider whether the response meets the user's length requirements in your evaluation. Ensure that only one integer between 1 and 5 is output for each dimension score."""
        image_path = get_image_path(idx, row)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": task_description.replace("$INST$", row['question']).replace("$RESPONSE$", row['prediction'])}
        ] + [{"type": "image_url", "image_url": {"image": encode_image_to_base64(p)}} for p in image_path]}]
        
        retry = 5
        use_cache = True
        for i in range(retry):
            try:
                res = GPT_Interface.call(model="gpt-4o", messages=messages, use_cache=use_cache, temperature=0.8)
                print(res)
                res_json = extract_json(res)
                for d in dims:
                    if d in res_json and res_json[d] in range(1, 6):
                        continue
                    else:
                        raise ValueError(f"Invalid score for dimension {d}: {res_json[d]}")
                return res_json
            except Exception as e:
                print(f"GPT API call failed due to {e}, retrying... ({i+1}/{retry})")
                if i == retry - 1:  # Last retry
                    raise e
                use_cache = False
                continue
    
    def evaluate_single(self, idx, row):
        """Evaluate a single prediction based on multiple metrics"""
        length = count_words(row['prediction'])
        l_score = length_score(length, row['L'])
        quality_res = self.evaluate_quality(idx, row)
        overall_quality_score = 0
        for d in dims:
            overall_quality_score += quality_res[d]
        
        overall_quality_score = (overall_quality_score - 6) * 25 / 6
        
        return {
            'length': length,
            'length_score': l_score,
            'quality_scores': quality_res,
            'relevance': quality_res['Relevance'],
            'accuracy': quality_res['Accuracy'],
            'coherence': quality_res['Coherence'],
            'clarity': quality_res['Clarity'],
            'breadth_depth': quality_res['Breadth and Depth'],
            'reading_experience': quality_res['Reading Experience'],
            'overall_quality_score': overall_quality_score
        }
    
    def run(self):
        """Run the complete evaluation pipeline"""
        # First pass: Generate predictions
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            if row['prediction'] is not None and not pd.isna(row['prediction']):
                continue
                
            res = self.predict(idx, row)
            print(res, flush=True)
            self.data.loc[idx, 'prediction'] = str(res)
            self.data.to_excel(self.output_path, index=False)
            
        self.data.to_excel(self.output_path, index=False)
        
        # Second pass: Evaluate predictions
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            if row['quality_scores'] is not None and not pd.isna(row['quality_scores']):
                continue
            
            res = self.evaluate_single(idx, row)
            for col, value in res.items():
                if col != 'quality_scores':
                    self.data.loc[idx, col] = float(value)
                else:
                    self.data.loc[idx, col] = str(value)
            self.data.to_excel(self.output_path, index=False)
        
        self.data.to_excel(self.output_path, index=False)
        return self.data
    
    def __del__(self):
        """Cleanup when the object is destroyed"""
        if self.model is not None:
            del self.model

def print_res_head():
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{S} & \\textbf{S_l} & \\textbf{S_q} & \\textbf{S_l0} & \\textbf{S_q0} & \\textbf{S_l1} & \\textbf{S_q1} & \\textbf{S_l2} & \\textbf{S_q2} & \\textbf{S_l3} & \\textbf{S_q3} \\\\")
    print("\\hline")

def print_res(model_res):
    format = """\\texttt{{{model}}} & {S:.1f} & {Sl:.1f} & {Sq:.1f} & {Sl_0:.1f} & {Sq_0:.1f} & {Sl_1:.1f} & {Sq_1:.1f} & {Sl_2:.1f} & {Sq_2:.1f} & {Sl_3:.1f} & {Sq_3:.1f} \\\\"""
    for m in model_res:
        df = pd.read_excel(model_res[m])
        Sl_group = [0, 0, 0, 0]
        Sq_group = [0, 0, 0, 0]
        cnt_group = [0, 0, 0, 0]
        for idx, row in df.iterrows():
            Sl = row['length_score']
            Sq = row['overall_quality_score']
            L = row['L']
            if L in range(0, 1500):
                Sl_group[0] += Sl
                Sq_group[0] += Sq
                cnt_group[0] += 1
            elif L in range(1500, 2000):
                Sl_group[1] += Sl
                Sq_group[1] += Sq
                cnt_group[1] += 1
            elif L in range(2000, 3000):
                Sl_group[2] += Sl
                Sq_group[2] += Sq
                cnt_group[2] += 1
            else:
                Sl_group[3] += Sl
                Sq_group[3] += Sq
                cnt_group[3] += 1
        
        Sl_all = sum(Sl_group) / sum(cnt_group)
        Sq_all = sum(Sq_group) / sum(cnt_group)
        S = (Sl_all + Sq_all) / 2
        
        # Combine the two dictionaries before passing to format
        format_dict = {
            "model": m,
            "S": S,
            "Sl": Sl_all,
            "Sq": Sq_all
        }
        format_dict.update({f"Sl_{i}": Sl_group[i] / cnt_group[i] for i in range(4)})
        format_dict.update({f"Sq_{i}": Sq_group[i] / cnt_group[i] for i in range(4)})
        print(format.format(**format_dict))


def length_score(x, y):
    """Calculate length score based on target length y and actual length x"""
    x = max(x, 2)
    y = max(y, 2)
    if y > x:
        return 100 * max(0, 1. - (y / x - 1) / 3)
    else:
        return 100 * max(0, 1. - (x / y - 1) / 2) 


if __name__ == "__main__":
    caption_llm = {
        "GLM-4-9B-Chat": "data/MMLongBench_Write_Caption_LLM_glm-4-9b.xlsx",
        "GPT-4o-2024-05-13": "data/MMLongBench_Write_Caption_LLM_gpt-4o.xlsx",
        "Mistral-Large-Instruct-2407": "data/MMLongBench_Write_Caption_LLM_mistral-large-instruct-2407.xlsx"
    }
    model_res = {
        "Qwen2-VL-7B": "data/MMLongBench_Write_VLM_Qwen2-VL-7B.xlsx",
        "Qwen2-VL-72B": "data/MMLongBench_Write_VLM_Qwen2-VL-72B.xlsx",
        "MiniCPM-V2.6": "data/MMLongBench_Write_VLM_MiniCPM-V-2_6.xlsx",
        "gpt-4-1106-vision-preview": "data/MMLongBench_Write_VLM_gpt-4-1106-vision-preview.xlsx",
        "gpt-4o-2024-05-13": "data/MMLongBench_Write_VLM_gpt-4o.xlsx",
        "claude-3-opus-20240229": "data/MMLongBench_Write_VLM_claude-3-opus-20240229.xlsx",
        "LongWriter-V-7B": "data/MMLongBench_Write_VLM_LongWriter-V-7B.xlsx",
        "LongWriter-V-72B": "data/MMLongBench_Write_VLM_LongWriter-V-72B.xlsx",
        "LongWriter-V-7B-DPO": "data/MMLongBench_Write_VLM_LongWriter-V-7B-DPO-Lec.xlsx",
        "Qwen2.5-VL-7B": "data/MMLongBench_Write_VLM_qwen2.5-vl-7b.xlsx",
        "Qwen2.5-VL-72B": "data/MMLongBench_Write_VLM_qwen2.5-vl-72b.xlsx",
    }
    ablation_res = {
        'Without Multi-Image': "data/MMLongBench_Write_VLM_ablation-single_image.xlsx",
        'Without Single-Image': "data/MMLongBench_Write_VLM_ablation-multi_image.xlsx",
    }
    print_res_head()
    print_res(ablation_res)