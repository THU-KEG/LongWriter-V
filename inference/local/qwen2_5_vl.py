import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from inference.local.base import BaseModel
from qwen_vl_utils import process_vision_info
import json

class Qwen2_5_VL(BaseModel):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path)
        self.load_kwargs = kwargs

    def _load(self):
        if self.model is None or self.processor is None:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, 
                device_map='auto', 
                attn_implementation='flash_attention_2', 
                torch_dtype=torch.bfloat16
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                device_map='auto',
                min_pixels=self.load_kwargs.get("min_pixels", 50176), 
                max_pixels=self.load_kwargs.get("max_pixels", 1048576)
            )
    
    def inference(self, msgs, **kwargs):
        self._load()
        
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(msgs)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        generated_ids = self.model.generate(
            **inputs,
            **kwargs
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0]


def get_model(type):
    with open("config.json", "r") as f:
        config = json.load(f)
    model_paths = config["model_paths"]["qwen2_5_vl"]
    return Qwen2_5_VL(model_paths[type])


if __name__ == "__main__":
    msgs = [{"role": "user", "content": "你好，你是谁？"}]
    model = get_model("7b")
    print(model.inference(msgs))
