import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from inference.local.base import BaseModel
from config import config

class GLM4(BaseModel):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path)
        self.load_kwargs = kwargs

    def _load(self):
        if self.model is None or self.processor is None:
            self.model = LLM(
                model=self.model_path,
                tensor_parallel_size=self.load_kwargs.get("tensor_parallel_size", 4),
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
    
    def inference(self, msgs, **kwargs):
        pass
            
    def inference_vllm(self, msgs, **kwargs):
        self._load()

        stop_token_ids = [151329, 151336, 151338]
        sampling_params = SamplingParams(
            **kwargs,
            stop_token_ids=stop_token_ids
       )

        inputs = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        outputs = self.model.generate(prompts=inputs, sampling_params=sampling_params)
        
        return outputs[0].outputs[0].text


def get_model(type):
    model_paths = config.model_paths["glm4"]
    return GLM4(model_paths[type])
