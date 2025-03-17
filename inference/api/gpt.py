import json
import requests
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pymongo import MongoClient
import time
import hashlib

from config import config

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
    
    def delete(self, key: str):
        self.collection.delete_one({"_id": key})

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

# Initialize cache with error handling
try:
    cache = MongoCache(
        host=config.mongo_cache_host,
        port=config.mongo_cache_port,
        db_name=config.mongo_cache_db
    )
except Exception as e:
    print(f"Warning: MongoDB cache initialization failed: {str(e)}")
    cache = None

class GPT_Interface:
    client = OpenAI(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url
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
        # Skip cache if MongoDB is not available
        if cache is None:
            use_cache = False
            
        if use_cache:
            cache_key = cls._generate_cache_key(model, messages, **kwargs)
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
        retries = 3
        delay = 1.0
        for attempt in range(retries):
            try:
                response = cls.client.chat.completions.create(model=model, messages=messages, **kwargs)
                if response.choices[0].message.refusal:
                    raise Exception("GPT refused to answer: " + response.choices[0].message.refusal)
                result = response.choices[0].message.content
                
                if use_cache and cache is not None:
                    cache_key = cls._generate_cache_key(model, messages, **kwargs)
                    cache[cache_key] = result
                return result
                
            except Exception as e:
                if attempt == retries - 1 or "GPT refused to answer" in str(e):
                    print(f"{model} API call failed: {str(e)}")
                    raise e
                else:
                    print(f"Attempt {attempt + 1} failed, caused by {str(e)}, retrying in {delay} seconds...")
                    time.sleep(delay)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached responses"""
        if cache is not None:
            cache.clear()

class DeepSeek_Interface(GPT_Interface):
    client = OpenAI(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url
    )

    @classmethod
    def call_with_reasoning(cls, model, messages, use_cache=True, **kwargs):
        """
        Common helper method for GPT API calls with caching and retry logic
        Returns: Tuple of (response_content, reasoning_content)
        """
        if cache is None:
            use_cache = False
            
        if use_cache:
            cache_key = cls._generate_cache_key(model, messages, **kwargs)
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs           
        }
        headers = {
            "Authorization": "Bearer " + config.deepseek_api_key,
            "Content-Type": "application/json"
        }
        retries = 3
        delay = 1.0
        for attempt in range(retries):
            try:
                response = requests.post(config.deepseek_base_url + "/chat/completions", headers=headers, json=payload)
                print(response.text)
                response = response.json()
                result = response["choices"][0]["message"]["content"]
                reasoning = response["choices"][0]["message"]["reasoning_content"]
                
                if use_cache and cache is not None:
                    cache_key = cls._generate_cache_key(model, messages, **kwargs)
                    cache[cache_key] = (result, reasoning)
                return result, reasoning
                
            except Exception as e:
                if attempt == retries - 1 or "GPT refused to answer" in str(e):
                    print(f"{model} API call failed: {str(e)}")
                    raise e
                else:
                    print(f"Attempt {attempt + 1} failed, caused by {str(e)}, retrying in {delay} seconds...")
                    time.sleep(delay)

class VllmServer_Interface(GPT_Interface):
    client = OpenAI(
        api_key=config.vllm_api_key,
        base_url=config.vllm_base_url
    )

class Gemini_Interface(GPT_Interface):
    client = OpenAI(
        api_key=config.gemini_api_key,
        base_url=config.gemini_base_url
    )

class Qwen_Interface(GPT_Interface):
    client = OpenAI(
        api_key=config.qwen_api_key,
        base_url=config.qwen_base_url
    )

class GLM_Interface(GPT_Interface):
    client = OpenAI(
        api_key=config.glm_api_key,
        base_url=config.glm_base_url
    )