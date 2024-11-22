import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from diskcache import Cache

class GPT_Interface:
    """Interface for calling OpenAI GPT-4 models with response caching"""
    
    # Initialize cache in project root directory
    cache_dir = Path(__file__).parent.parent / ".cache"
    try:
        cache = Cache(str(cache_dir), timeout=1, disk_min_file_size=0)
    except Exception as e:
        print(f"Warning: Cache initialization failed: {str(e)}")
        # Fallback to a temporary directory if main cache fails
        import tempfile
        temp_cache_dir = Path(tempfile.gettempdir()) / ".cache_fallback"
        cache = Cache(str(temp_cache_dir), timeout=1, disk_min_file_size=0)
    
    # Load config file
    config_path = Path(__file__).parent.parent / "config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
            client = OpenAI(
                api_key=config["api_key"],
                base_url=config.get("base_url")
            )
    except Exception as e:
        raise Exception(f"Failed to load config file: {str(e)}")

    @classmethod
    def _call_gpt(cls, model: str, messages: List[Dict[str, Any]], 
                  temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """
        Common helper method for GPT API calls with caching
        """
        cache_key = f"{model}_{json.dumps(messages)}_{temperature}_{max_tokens}"
        
        if cache_key in cls.cache:
            return cls.cache[cache_key]
            
        try:
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                params["max_tokens"] = max_tokens
                
            response = cls.client.chat.completions.create(**params)
            result = response.choices[0].message.content
            
            cls.cache[cache_key] = result
            return result
            
        except Exception as e:
            raise Exception(f"{model} API call failed: {str(e)}")

    @classmethod
    def call_gpt4o(cls, messages: List[Dict[str, str]], 
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None) -> str:
        return cls._call_gpt("gpt-4o", messages, temperature, max_tokens)
    
    @classmethod        
    def call_gpt4v(cls, messages: List[Dict[str, Any]],
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None) -> str:
        return cls._call_gpt("gpt-4-vision-preview", messages, temperature, 
                           max_tokens or 4096)
