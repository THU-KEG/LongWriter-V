import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional, List, Tuple, Dict
import PIL

class ModelManager:
    def __init__(self):
        self.reranker_model = None
        self.reranker_tokenizer = None
        self.rm_model = None
        self.rm_tokenizer = None

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

    def inference_reranker(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Run inference for reranker model to score query-candidate pairs"""
        model, tokenizer = self._load_reranker()
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        return scores.tolist()
    
    def inference_rm(self, msgs: List[str]) -> List[float]:
        model, tokenizer = self._load_rm()
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
        return answer