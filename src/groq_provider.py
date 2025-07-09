"""Groq LLM Provider - Fast Free API"""
import requests
import json
from typing import Dict, Any, Optional
from loguru import logger

class GroqProvider:
    """Groq API provider - very fast and has generous free tier"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        """Initialize Groq provider
        Free models: llama-3.1-8b-instant, llama-3.1-70b-versatile, mixtral-8x7b-32768, gemma-7b-it
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Test Groq API connection"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                models = response.json().get("data", [])
                available_models = [m["id"] for m in models]
                
                if self.model in available_models:
                    self.is_initialized = True
                    logger.info(f"Groq provider initialized with model: {self.model}")
                    return True
                else:
                    logger.error(f"Model {self.model} not available. Available: {available_models}")
                    return False
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Groq provider: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Groq API"""
        if not self.is_initialized:
            return "Error: Groq provider not initialized"
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
                "top_p": kwargs.get("top_p", 0.9)
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                error_msg = f"Groq API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
                
        except Exception as e:
            error_msg = f"Error generating with Groq: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": "groq",
            "model": self.model,
            "cost": "FREE TIER",
            "speed": "Very Fast",
            "privacy": "API-based"
        }
