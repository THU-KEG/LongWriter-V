import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pymongo import MongoClient
import time
import hashlib

config_path = "config.json"
config = json.load(open(config_path))

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

cache = MongoCache(
    host=config["mongo_cache_host"],
    port=config["mongo_cache_port"],
    db_name=config["mongo_cache_db"]
)

class GPT_Interface:
    client = OpenAI(
        api_key=config["openai_api_key"],
        base_url=config.get("openai_base_url")
    )

    @staticmethod
    def _generate_cache_key(model: str, messages: List[Dict[str, Any]], 
                           **kwargs) -> str:
        """Generate a stable, fixed-length cache key using SHA-256"""
        simplified_messages = [
            {
                'role': msg['role'],
                'content': msg['content'] if isinstance(msg['content'], str) 
                          else str(msg['content'])
            }
            for msg in messages
        ]
        key_content = f"{model}_{json.dumps(simplified_messages, sort_keys=True)}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.sha256(key_content.encode()).hexdigest()

    @classmethod
    def call(cls, model, messages, use_cache=True, **kwargs):
        """
        Common helper method for GPT API calls with caching and retry logic
        Returns: Tuple of (response_content, prompt_tokens, completion_tokens)
        """
        cache_key = cls._generate_cache_key(model, messages, **kwargs)
        
        if use_cache:
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
        retries = 3
        delay = 1.0
        for attempt in range(retries):
            try:
                response = cls.client.chat.completions.create(model=model, messages=messages, **kwargs)

                result = response.choices[0].message.content
                
                if use_cache:
                    cache[cache_key] = result
                return result
                
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Attempt {attempt + 1} failed, caused by {str(e)}, retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"{model} API call failed after {retries} attempts: {str(e)}")
                    raise e

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached responses"""
        cache.clear()

class DeepSeek_Interface(GPT_Interface):
    client = OpenAI(
        api_key=config["deepseek_api_key"],
        base_url=config.get("deepseek_base_url")
    )
        

if __name__ == "__main__":
    # GPT_Interface.clear_cache()
    messages = [
        {"role": "user", "content": "Hello, who are you?"}
    ]
    print(DeepSeek_Interface.call(model="deepseek-reasoner", messages=messages, use_cache=False))