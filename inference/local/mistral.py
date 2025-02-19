from vllm import LLM, SamplingParams
from inference.local.base import BaseModel
from config import config

class Mistral(BaseModel):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path)
        self.load_kwargs = kwargs
    
    def _load(self):
        if self.model is None:
            self.model = LLM(
                model=self.model_path,
                tensor_parallel_size=self.load_kwargs.get("tensor_parallel_size", 8),
                trust_remote_code=True
            )

    def inference(self, msgs, **kwargs):
        self._load()

        sampling_params = SamplingParams(
            **kwargs
        )

        res = self.model.chat(msgs, sampling_params)
        return res[0].outputs[0].text

def get_model(type):
    model_paths = config.model_paths["mistral"]
    return Mistral(model_paths[type])