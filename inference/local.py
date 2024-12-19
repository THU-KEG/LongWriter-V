import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Optional, List, Tuple, Dict
import PIL

class ModelManager:
    def __init__(self):
        self.reranker_model = None
        self.reranker_tokenizer = None
        self.rm_model = None
        self.rm_tokenizer = None
        self.qwen2_vl_7b_model = None
        self.qwen2_vl_7b_tokenizer = None
        self.qwen2_vl_72b_model = None
        self.qwen2_vl_72b_tokenizer = None

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
    
    def _load_qwen2_vl(self, type: str = '7b'):
        if type == '7b':
            if self.qwen2_vl_7b_model is None or self.qwen2_vl_7b_tokenizer is None:
                self.qwen2_vl_7b_model = Qwen2VLForConditionalGeneration.from_pretrained('/model/inference/Qwen/Qwen2-VL-7B-Instruct', trust_remote_code=True, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
                self.qwen2_vl_7b_tokenizer = AutoProcessor.from_pretrained('/model/inference/Qwen/Qwen2-VL-7B-Instruct', trust_remote_code=True)
        elif type == '72b':
            if self.qwen2_vl_72b_model is None or self.qwen2_vl_72b_tokenizer is None:
                self.qwen2_vl_72b_model = Qwen2VLForConditionalGeneration.from_pretrained('/model/inference/Qwen/Qwen2-VL-72B-Instruct', trust_remote_code=True, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
                self.qwen2_vl_72b_tokenizer = AutoProcessor.from_pretrained('/model/inference/Qwen/Qwen2-VL-72B-Instruct', trust_remote_code=True)
        return self.qwen2_vl_7b_model, self.qwen2_vl_7b_tokenizer

    def inference_qwen2_vl(self, msgs, type='7b'):
        model, tokenizer = self._load_qwen2_vl(type)
        inputs = tokenizer(msgs, return_tensors='pt')
        outputs = model.generate(**inputs, max_new_tokens=8192)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def inference(self, msgs, type='gpt-4o'):
        if type == 'gpt-4o':
            return self.inference_gpt4o(msgs)
        elif type == 'qwen2-vl':
            return self.inference_qwen2_vl(msgs)
        else:
            raise ValueError(f"Invalid model type: {type}")