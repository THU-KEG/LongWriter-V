import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pymongo import MongoClient
import time
import hashlib

class MongoCache:
    """MongoDB-based cache implementation"""
    def __init__(self, host: str, port: int, db_name: str):
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.collection = self.db['api_cache']
        # Create TTL index on created_at field
        self.collection.create_index("created_at", expireAfterSeconds=7*24*60*60)  # 7 day TTL

    def get(self, key: str) -> Optional[tuple]:
        result = self.collection.find_one({"_id": key})
        return result["value"] if result else None

    def __setitem__(self, key: str, value: tuple):
        self.collection.update_one(
            {"_id": key},
            {
                "$set": {
                    "value": value,
                    "created_at": time.time()
                }
            },
            upsert=True
        )

    def clear(self):
        self.collection.delete_many({})

class GPT_Interface:
    """Interface for calling OpenAI GPT-4 models with response caching"""
    
    # Load config file
    config_path = "config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
            client = OpenAI(
                api_key=config["api_key"],
                base_url=config.get("base_url")
            )
            # Initialize MongoDB cache
            cache = MongoCache(
                host=config["mongo_cache_host"],
                port=config["mongo_cache_port"],
                db_name=config["mongo_cache_db"]
            )
    except Exception as e:
        raise Exception(f"Failed to load config or initialize cache: {str(e)}")

    @staticmethod
    def _generate_cache_key(model: str, messages: List[Dict[str, Any]], 
                           temperature: float, max_tokens: Optional[int]) -> str:
        """Generate a stable, fixed-length cache key using SHA-256"""
        # Create a more stable message representation
        simplified_messages = [
            {
                'role': msg['role'],
                'content': msg['content'] if isinstance(msg['content'], str) 
                          else str(msg['content'])  # Handle non-string content
            }
            for msg in messages
        ]
        # Create a string combining all relevant parameters
        key_content = f"{model}_{json.dumps(simplified_messages, sort_keys=True)}_{temperature}_{max_tokens}"
        # Generate SHA-256 hash
        return hashlib.sha256(key_content.encode()).hexdigest()

    @classmethod
    def _call_gpt(cls, model: str, messages: List[Dict[str, Any]], 
                  temperature: float = 0.7, max_tokens: Optional[int] = None,
                  retries: int = 3, delay: float = 1.0, use_cache: bool = True) -> tuple[str, int, int]:
        """
        Common helper method for GPT API calls with caching and retry logic
        Returns: Tuple of (response_content, prompt_tokens, completion_tokens)
        """
        cache_key = cls._generate_cache_key(model, messages, temperature, max_tokens)
        
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
    def call_gpt4o_0513(cls, messages: List[Dict[str, str]], 
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None,
                   use_cache: bool = True) -> tuple[str, int, int]:
        return cls._call_gpt("gpt-4o-2024-05-13", messages, temperature, max_tokens, use_cache=use_cache)
    
    @classmethod
    def call_gpt4o_mini(cls, messages: List[Dict[str, str]], 
                        temperature: float = 0.7,
                        max_tokens: Optional[int] = None,
                        use_cache: bool = True) -> tuple[str, int, int]:
        return cls._call_gpt("gpt-4o-mini", messages, temperature, max_tokens, use_cache=use_cache)
    
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