import os
import pandas as pd
from utils import count_words, extract_json, encode_image_to_base64, parallel_process
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
    def __init__(self, output_path, data_path=None):
        self.output_path = output_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize the data DataFrame with required columns"""
        if os.path.exists(self.output_path):
            self.data = pd.read_excel(self.output_path)
        elif self.data_path:
            self.data = pd.read_excel(self.data_path)
        else:
            # Child class should override this method if not using Excel files
            self.data = pd.DataFrame()
        
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
        task_description = open('prompts/eval_quality.txt', 'r').read()
        image_path = get_image_path(idx, row)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": task_description.replace("$INST$", row['question']).replace("$RESPONSE$", row['prediction'])}
        ] + [{"type": "image_url", "image_url": {"url": encode_image_to_base64(p)}} for p in image_path]}]
        
        retry = 5
        use_cache = True
        for i in range(retry):
            try:
                res = GPT_Interface.call(model="gpt-4o-2024-08-06", messages=messages, use_cache=use_cache, temperature=0.5, max_tokens=1024)
                print(res)
                res_json = extract_json(res)
                for d in dims:
                    if d in res_json and res_json[d] in range(1, 6):
                        continue
                    else:
                        raise ValueError(f"Invalid score for dimension {d}: {res_json[d]}")
                return res_json
            except Exception as e:
                if "GPT refused to answer" in str(e):
                    print(f"GPT refused to answer for index {idx}, skipping evaluation")
                    return None
                print(f"GPT API call failed due to {e}, retrying... ({i+1}/{retry})")
                if i == retry - 1:  # Last retry
                    print(f"All retries failed for index {idx}")
                    raise e
                use_cache = False
                continue
    
    def evaluate_single(self, idx, row):
        """Evaluate a single prediction based on multiple metrics"""
        length = count_words(row['prediction'])
        l_score = length_score(length, row['L'])
        quality_res = self.evaluate_quality(idx, row)
        
        # Handle case where quality evaluation was skipped
        if quality_res is None:
            return {
                'length': length,
                'length_score': l_score,
                'quality_scores': 'GPT_REFUSED',  # Changed to string marker
                'relevance': None,
                'accuracy': None,
                'coherence': None,
                'clarity': None,
                'breadth_depth': None,
                'reading_experience': None,
                'overall_quality_score': None
            }
        
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
        """Run the complete evaluation pipeline with parallel processing"""
        # First pass: Generate predictions
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            if row['prediction'] is not None and not pd.isna(row['prediction']):
                continue
                
            res = self.predict(idx, row)
            print(res, flush=True)
            self.data.loc[idx, 'prediction'] = str(res)
            self.data.to_excel(self.output_path, index=False)
            
        self.data.to_excel(self.output_path, index=False)
        
        # Second pass: Evaluate predictions in parallel
        evaluation_batch = []
        for idx, row in self.data.iterrows():
            if row['quality_scores'] is not None and not pd.isna(row['quality_scores']):
                continue
            evaluation_batch.append((idx, row))
        
        if evaluation_batch:
            # Process evaluations in parallel
            batch_results = parallel_process(
                self.evaluate_single,
                evaluation_batch,
                num_processes=5
            )
            
            # Update results one by one
            for (idx, _), res in zip(evaluation_batch, batch_results):
                if res:
                    for col, value in res.items():
                        if col == 'quality_scores':
                            self.data.loc[idx, col] = str(value)
                        else:
                            # Handle None values for numeric columns
                            if value is None:
                                self.data.loc[idx, col] = pd.NA
                            else:
                                self.data.loc[idx, col] = float(value)
        
        self.data.to_excel(self.output_path, index=False)
        return self.data

    def __del__(self):
        """Cleanup when the object is destroyed"""
        if self.model is not None:
            del self.model

def print_res(model_res):
    format = """\\texttt{{{model}}} & {S:.1f} & {Sl:.1f} & {Sq:.1f} & {Sl_0:.1f} & {Sq_0:.1f} & {Sl_1:.1f} & {Sq_1:.1f} & {Sl_2:.1f} & {Sq_2:.1f} & {Sl_3:.1f} & {Sq_3:.1f} \\\\"""
    for m in model_res:
        df = pd.read_excel(model_res[m])
        Sl_group = [0, 0, 0, 0]
        Sq_group = [0, 0, 0, 0]
        cnt_group = [0, 0, 0, 0]
        
        # Filter out rows where GPT refused to answer
        df = df[df['quality_scores'] != 'GPT_REFUSED']
        
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
        
        # Check if we have any valid data points
        if sum(cnt_group) == 0:
            print(f"Warning: No valid evaluations for model {m}")
            continue
            
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
        # Handle potential division by zero for each group
        for i in range(4):
            format_dict[f"Sl_{i}"] = Sl_group[i] / cnt_group[i] if cnt_group[i] > 0 else 0
            format_dict[f"Sq_{i}"] = Sq_group[i] / cnt_group[i] if cnt_group[i] > 0 else 0
            
        print(format.format(**format_dict))


def length_score(x, y):
    """Calculate length score based on target length y and actual length x"""
    x = max(x, 2)
    y = max(y, 2)
    if y > x:
        return 100 * max(0, 1. - (y / x - 1) / 3)
    else:
        return 100 * max(0, 1. - (x / y - 1) / 2) 