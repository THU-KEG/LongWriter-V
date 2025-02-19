from inference.local.base import BaseModel
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor
)
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
import json


class Qwen2VL(BaseModel):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path)
        self.load_kwargs = kwargs

    def _load(self):
        if self.model is None or self.processor is None:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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
    
    def _load_vllm(self):
        if self.model is None or self.processor is None:
            self.model = LLM(
                model=self.model_path, 
                trust_remote_code=True, 
                tensor_parallel_size=self.load_kwargs.get("tensor_parallel_size", 4),
                limit_mm_per_prompt={"image":26}
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=self.load_kwargs.get("min_pixels", 50176),
                max_pixels=self.load_kwargs.get("max_pixels", 1048576)
            )
    
    def _load_lora(self, lora_path):
        if self.model is None or self.processor is None:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map='cuda:0',
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                device_map='cuda:0',
                min_pixels=self.load_kwargs.get("min_pixels", 50176),
                max_pixels=self.load_kwargs.get("max_pixels", 1048576)
            )
            self.model = PeftModel.from_pretrained(self.model, lora_path, device_map="cuda:0")
            self.model = self.model.merge_and_unload()
            lora_params = [n for n, _ in self.model.named_parameters() if 'lora' in n]
            print(f"Found {len(lora_params)} LoRA parameters")
    
    def inference_lora(self, msgs, lora_path, **kwargs):
        self._load_lora(lora_path)
       
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
    
        # Move all input tensors to cuda:0
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
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
            clean_up_tokenization_spaces=True
        )
        return output_text[0]
    
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
            clean_up_tokenization_spaces=True
        )
        
        return output_text[0]
    
    def inference_vllm(self, msgs, **kwargs):
        self._load_vllm()

        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(msgs)

        sampling_params = SamplingParams(
            **kwargs
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

        outputs = self.model.generate([llm_inputs], sampling_params=sampling_params)
        
        # Extract all generated texts from each output
        all_texts = []
        for output in outputs:
            all_texts.extend([gen.text for gen in output.outputs])
        
        return all_texts


def get_model(type, **kwargs):
    with open("config.json", "r") as f:
        config = json.load(f)
    model_paths = config["model_paths"]["qwen2_vl"]
    return Qwen2VL(model_paths[type], **kwargs)
