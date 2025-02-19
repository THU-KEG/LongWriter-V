from pathlib import Path

BASE_DIR = Path(__file__).parent

class Config:
    def __init__(self):
        # Default configuration values
        self.openai_api_key = ""
        self.openai_base_url = "https://api.openai.com/v1/"
        self.deepseek_api_key = ""
        self.deepseek_base_url = ""
        self.mongo_cache_host = "localhost"
        self.mongo_cache_port = 27017
        self.mongo_cache_db = "api_cache"
        self.vllm_api_key = ""
        self.vllm_base_url = ""
        self.claude_api_key = ""
        self.claude_base_url = ""
        self.gemini_api_key = ""
        self.gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.model_paths = {
            "clip": {
                "base": "openai/clip-vit-base-patch32"
            },
            "qwen2_vl": {
                "7b": "Qwen/Qwen2-VL-7B-Instruct",
                "72b": "Qwen/Qwen2-VL-72B-Instruct",
            },
            "qwen2_5_vl": {
                "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
                "72b": "Qwen/Qwen2.5-VL-72B-Instruct",
                "longwriter-v-7b": "THU-KEG/LongWriter-V-7B",
                "longwriter-v-7b-dpo": "THU-KEG/LongWriter-V-7B-DPO",
                "longwriter-v-72b": "THU-KEG/LongWriter-V-72B",
            },
            "glm4": {
                "9b-chat": "THUDM/glm-4-9b-chat"
            },
            "mistral": {
                "large-instruct-2407": "mistralai/Mistral-Large-Instruct-2407"
            },
            "rm": {
                "base": ""
            },
            "reranker": {
                "base": ""
            },
            "minicpm": {
                "base": "openbmb/MiniCPM-V-2_6"
            }
        }

config = Config()