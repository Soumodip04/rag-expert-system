"""Ollama LLM Provider - Free Local Models"""
import requests
import json
from typing import Dict, Any, Optional
from loguru import logger

class OllamaProvider:
    """Ollama local LLM provider - completely free"""
    
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize Ollama connection"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                
                if self.model not in available_models:
                    print(f"Model {self.model} not found. Available models: {available_models}")
                    print(f"Run: ollama pull {self.model}")
                    return False
                    
                self.is_initialized = True
                return True
            else:
                print("Ollama not running. Please start Ollama first.")
                return False
                
        except requests.exceptions.ConnectionError:
            print("Cannot connect to Ollama. Please install and start Ollama.")
            print("Download from: https://ollama.ai")
            return False
        except Exception as e:
            print(f"Error initializing Ollama: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama"""
        if not self.is_initialized:
            return "Error: Ollama not initialized"
            
        try:
            # Prepare request
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "top_p": kwargs.get("top_p", 0.9)
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Error: Ollama request failed with status {response.status_code}"
                
        except Exception as e:
            return f"Error generating with Ollama: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "ollama",
            "model": self.model,
            "cost": "FREE",
            "privacy": "100% Local"
        }
