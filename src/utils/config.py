import os

# API Keys and Endpoints for different LLM providers

CONFIG = {
    "gemini": {
        "api_key": "AIzaSyDdkiVH8M-P_MpWbO4tAH88t2MLOemR9XY",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models"
    },
    "openai": {
        "api_key": "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",
        "endpoint": "https://api.openai.com/v1"
    },
    "groq": {
        "api_key": "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",
        "endpoint": "https://api.groq.com/v1"
    },
    "groq2": {
        "api_key": "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",
        "endpoint": "https://api.groq.com/v1"
    },
    "openrouter": {
        "api_key": "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",
        "endpoint": "https://openrouter.ai/api/v1"
    }
}

def get_api_key(provider: str) -> str:
    """Get API key for a given provider."""
    return CONFIG.get(provider, {}).get("api_key", "")

def get_endpoint(provider: str) -> str:
    """Get endpoint for a given provider."""
    return CONFIG.get(provider, {}).get("endpoint", "")