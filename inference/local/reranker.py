import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from inference.local.base import BaseModel
from config import config

class Reranker(BaseModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        
    def _load(self):
        if self.model is None or self.processor is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.processor = AutoTokenizer.from_pretrained(self.model_path)
            self.model.eval()
    
    def inference(self, msgs):
        self._load()
        with torch.no_grad():
            inputs = self.processor(msgs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        return scores.tolist()
    
def get_model(type):
    model_paths = config.model_paths["reranker"]
    return Reranker(model_paths["base"])