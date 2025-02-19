from inference.local.base import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch
from typing import List, Dict
from config import config

class RM(BaseModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        
    def _load(self):
        if self.model is None or self.processor is None:
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
            self.model = self.model.eval().cuda()
            self.processor = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)

    def inference(self, msgs: List[Dict]):
        self._load()
        answer = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.processor
        )
        return answer

def get_model(type):
    model_paths = config.model_paths["rm"]
    return RM(model_paths["base"])