import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from diskcache import Cache
import time

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
                  temperature: float = 0.7, max_tokens: Optional[int] = None,
                  retries: int = 3, delay: float = 1.0, use_cache: bool = True) -> tuple[str, int, int]:
        """
        Common helper method for GPT API calls with caching and retry logic
        Returns: Tuple of (response_content, prompt_tokens, completion_tokens)
        """
        # Create a more stable cache key by only including essential message content
        simplified_messages = [
            {
                'role': msg['role'],
                'content': msg['content'] if isinstance(msg['content'], str) 
                          else str(msg['content'])  # Handle non-string content
            }
            for msg in messages
        ]
        cache_key = f"{model}_{json.dumps(simplified_messages, sort_keys=True)}_{temperature}_{max_tokens}"
        
        if use_cache:
            cached_result = cls.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
        for attempt in range(retries):
            try:
                params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "timeout": 600.0
                }
                if max_tokens:
                    params["max_tokens"] = max_tokens
                    
                response = cls.client.chat.completions.create(**params)

                print(response)

                result = (response.choices[0].message.content, 
                         response.usage.prompt_tokens,
                         response.usage.completion_tokens)
                
                if use_cache:
                    cls.cache[cache_key] = result
                return result
                
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Attempt {attempt + 1} failed, caused by {str(e)}, retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise Exception(f"{model} API call failed after {retries} attempts: {str(e)}")

    @classmethod
    def call_gpt4o(cls, messages: List[Dict[str, str]], 
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None,
                   use_cache: bool = True) -> tuple[str, int, int]:
        return cls._call_gpt("gpt-4o", messages, temperature, max_tokens, use_cache=use_cache)
    
    @classmethod        
    def call_gpt4v(cls, messages: List[Dict[str, Any]],
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None,
                   use_cache: bool = True) -> tuple[str, int, int]:
        return cls._call_gpt("gpt-4-vision-preview", messages, temperature, 
                           max_tokens or 4096, use_cache=use_cache)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached responses"""
        cls.cache.clear()

if __name__ == "__main__":
    GPT_Interface.clear_cache()