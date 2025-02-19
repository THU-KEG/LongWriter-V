import json
from inference.local.base import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch

class MiniCPM(BaseModel):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path)
        self.load_kwargs = kwargs
    
    def _load(self):
        if self.model is None or self.processor is None:
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16).eval().cuda()
            self.processor = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def inference(self, msgs, **kwargs):
        self._load()
        return self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.processor,
            max_inp_length=8192,
            use_image_id=True,
            **kwargs
        )

def get_model(type, **kwargs):
    with open("config.json", "r") as f:
        config = json.load(f)
    model_paths = config["model_paths"]["minicpm"]
    return MiniCPM(model_paths[type], **kwargs)