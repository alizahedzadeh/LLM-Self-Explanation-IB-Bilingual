import os
import json
import requests
import time
from typing import Dict, List, Any, Optional, Union
import tiktoken

class LLMClient:
    """Base class for LLM API clients"""
    def __init__(self):
        self.last_call_time = 0
        self.rate_limit_delay = 1  # Default 1 second between calls

    def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_call_time = time.time()

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion from prompt"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text"""
        raise NotImplementedError("Subclasses must implement this method")


class OpenRouterClient(LLMClient):
    """Client for accessing Llama 4 through OpenRouter"""
    
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3-opus", rate_limit_delay: int = 1):
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key must be provided or set as OPENROUTER_API_KEY env var")
        self.model = model
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        # Use cl100k_base tokenizer (similar to most modern models)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: int = 1000,
                stop_sequences: List[str] = None,
                get_logprobs: bool = False,
                **kwargs) -> Dict[str, Any]:
        """Generate completion from Llama 4 via OpenRouter"""
        self._rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if stop_sequences:
            data["stop"] = stop_sequences
            
        if get_logprobs:
            data["logprobs"] = True
            data["top_logprobs"] = 5
        
        response = requests.post(self.base_url, headers=headers, json=data)
        
        if response.status_code != 200:
            error_message = f"OpenRouter API error: {response.status_code} - {response.text}"
            print(error_message)
            return {"error": error_message}
        
        result = response.json()
        
        # Extract the relevant parts of the response
        completion = {
            "text": result["choices"][0]["message"]["content"],
            "finish_reason": result["choices"][0]["finish_reason"],
            "model": result["model"],
        }
        
        if get_logprobs and "logprobs" in result["choices"][0]:
            completion["logprobs"] = result["choices"][0]["logprobs"]
        
        return completion
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text"""
        return len(self.tokenizer.encode(text))


class GPT4OMiniClient(LLMClient):
    """Client for accessing GPT-4o-mini"""
    
    def __init__(self, api_key: str = None, rate_limit_delay: int = 1):
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY env var")
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://api.openai.com/v1/chat/completions"
        # Use cl100k_base tokenizer (for GPT models)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: int = 1000,
                stop_sequences: List[str] = None,
                get_logprobs: bool = False,
                **kwargs) -> Dict[str, Any]:
        """Generate completion from GPT-4o-mini"""
        self._rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        data = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if stop_sequences:
            data["stop"] = stop_sequences
            
        if get_logprobs:
            data["logprobs"] = True
            data["top_logprobs"] = 5
        
        response = requests.post(self.base_url, headers=headers, json=data)
        
        if response.status_code != 200:
            error_message = f"OpenAI API error: {response.status_code} - {response.text}"
            print(error_message)
            return {"error": error_message}
        
        result = response.json()
        
        # Extract the relevant parts of the response
        completion = {
            "text": result["choices"][0]["message"]["content"],
            "finish_reason": result["choices"][0]["finish_reason"],
            "model": result["model"],
        }
        
        if get_logprobs and "logprobs" in result["choices"][0]:
            completion["logprobs"] = result["choices"][0]["logprobs"]
        
        return completion
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text"""
        return len(self.tokenizer.encode(text))