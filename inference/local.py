import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Optional, List, Tuple, Dict
import PIL
from qwen_vl_utils import process_vision_info

class ModelManager:
    def __init__(self):
        self.reranker_model: Optional['AutoModelForSequenceClassification'] = None
        self.reranker_tokenizer: Optional['AutoTokenizer'] = None
        self.rm_model: Optional['AutoModel'] = None
        self.rm_tokenizer: Optional['AutoTokenizer'] = None
        self.qwen2_vl_model: Optional['Qwen2VLForConditionalGeneration'] = None
        self.qwen2_vl_tokenizer: Optional['AutoProcessor'] = None
        self.vllm_model: Optional['LLM'] = None
        # Cache for loaded models
        self._loaded_model_type = None
        self._loaded_model = None
        self._loaded_processor = None

    def _load_reranker(self, model_path: str = '/model/inference/BAAI/bge-reranker-v2-m3'):
        if self.reranker_model is None or self.reranker_tokenizer is None:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.reranker_model.eval()
        return self.reranker_model, self.reranker_tokenizer
    
    def _load_rm(self, model_path: str = '/model/eval'):
        if self.rm_model is None or self.rm_tokenizer is None:
            self.rm_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
            self.rm_model = self.rm_model.eval().cuda()
            self.rm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
        return self.rm_model, self.rm_tokenizer

    def inference_reranker(self, pairs):
        """Run inference for reranker model to score query-candidate pairs"""
        model, tokenizer = self._load_reranker()
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        return scores.tolist()
    
    def inference_rm(self, msgs):
        model, tokenizer = self._load_rm()
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
        return answer
    
    def _load_qwen2_vl(self, model_path):
        min_pixels = 224*224
        max_pixels = 1024*1024
        if self.qwen2_vl_model is None or self.qwen2_vl_tokenizer is None:
            self.qwen2_vl_model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, device_map='auto', attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
            self.qwen2_vl_tokenizer = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        return self.qwen2_vl_model, self.qwen2_vl_tokenizer
    
    def _get_model_path(self, type: str) -> str:
        """Get the model path based on model type"""
        base_type = type.replace('-vllm', '')
        paths = {
            '7b': '/model/base/Qwen/Qwen2-VL-7B-Instruct',
            '72b': '/model/base/Qwen/Qwen2-VL-72B-Instruct',
            'longwriter-v': '/model/trained/Qwen/qwen2_vl-7b/inst_and_part_scripts_sample_10k_back_translated_5k',
            'longwriter-v-72b': '/model/trained/Qwen/qwen2_vl-72b/inst_and_part_scripts_sample_10k_back_translated_5k'
        }
        if base_type not in paths:
            raise ValueError(f"Invalid model type: {type}")
        return paths[base_type]

    def _load_model(self, type: str):
        # Return cached model if already loaded
        if type == self._loaded_model_type and self._loaded_model is not None:
            return self._loaded_model, self._loaded_processor

        model_path = self._get_model_path(type)
        
        # Load processor first since it's needed for both vllm and regular models
        self._loaded_processor = AutoProcessor.from_pretrained(model_path)
        
        # Load appropriate model type
        if type.endswith('-vllm'):
            # Import vllm only when needed
            from vllm import LLM, SamplingParams
            if self.vllm_model is None:
                self.vllm_model = LLM(model=model_path, limit_mm_per_prompt={"image": 26})
            self._loaded_model = self.vllm_model
        else:
            self._loaded_model, _ = self._load_qwen2_vl(model_path)
            
        self._loaded_model_type = type
        return self._loaded_model, self._loaded_processor

    def inference_qwen2_vl_vllm(self, msgs, type: str = '7b', num_samples: int = 1):
        """Run inference using Qwen2-VL model with vllm backend and support for multiple samples
        Args:
            msgs: List of message dictionaries containing the conversation
            type: Model type ('7b', '72b', 'longwriter-v', 'longwriter-v-72b')
            num_samples: Number of samples to generate
        Returns:
            A list of strings containing num_samples generated texts
        """
        # Add vllm suffix for model loading
        vllm_type = f"{type}-vllm"
        model, processor = self._load_model(vllm_type)
        
        # Import vllm for type checking
        from vllm import LLM, SamplingParams
        if not isinstance(model, LLM):
            raise ValueError("Model is not a vllm model")
            
        # Preparation for inference
        prompt = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(msgs)

        sampling_params = SamplingParams(
            max_tokens=8192,
            temperature=0.9,
            top_p=0.9,
            n=num_samples
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = model.generate([llm_inputs], sampling_params=sampling_params)
        
        # Extract all generated texts from each output
        all_texts = []
        for output in outputs:
            all_texts.extend([gen.text for gen in output.outputs])
        
        if len(all_texts) < num_samples:
            print(f"Warning: Only generated {len(all_texts)} samples out of requested {num_samples}")
        
        return all_texts

    def inference_qwen2_vl(self, msgs, type: str = '7b'):
        """Run inference using Qwen2-VL model
        Args:
            msgs: List of message dictionaries containing the conversation
            type: Model type ('7b', '72b', 'longwriter-v', 'longwriter-v-72b')
        Returns:
            A single string containing the generated text
        """
        model, processor = self._load_model(type)
       
        # Preparation for inference
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(msgs)
        
        # Add random seed for different results each time
        import time
        random_seed = int(time.time() * 1000) % 10000
        torch.manual_seed(random_seed)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8192,
            temperature=0.9,
            do_sample=True,
            top_p=0.9
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        
        return output_text[0]

if __name__ == "__main__":
    model_manager = ModelManager()
    # Example usage:
    msgs = [dict(role="user", content=[dict(type="text", text="今天天气怎么样？")])]
    # Using regular Qwen2-VL model:
    print(model_manager.inference_qwen2_vl(msgs, type='72b'))
    # Using Qwen2-VL with vllm backend and multiple samples:
    print(model_manager.inference_qwen2_vl_vllm(msgs, type='72b', num_samples=3))