"""LLM providers and generation pipeline"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from loguru import logger
import json
import os

# Import free local providers
try:
    from .ollama_provider import OllamaProvider
except ImportError:
    OllamaProvider = None

try:  
    from .huggingface_local_provider import HuggingFaceLocalProvider
except ImportError:
    HuggingFaceLocalProvider = None

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None

@dataclass
class GenerationResult:
    """Result from text generation"""
    text: str
    usage: Dict[str, int]
    model: str
    finish_reason: str
    metadata: Dict[str, Any] = None

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig = None) -> GenerationResult:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, config: GenerationConfig = None) -> Generator[str, None, None]:
        """Generate text as a stream"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI provider initialized with model: {self.model}")
        except ImportError:
            logger.error("OpenAI library not installed. Please install with: pip install openai")
            raise
    
    def generate(self, prompt: str, config: GenerationConfig = None) -> GenerationResult:
        """Generate text using OpenAI API"""
        if config is None:
            config = GenerationConfig()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop_sequences
            )
            
            return GenerationResult(
                text=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=response.model,
                finish_reason=response.choices[0].finish_reason
            )
        
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return GenerationResult(
                text=f"Error: {str(e)}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.model,
                finish_reason="error"
            )
    
    def generate_stream(self, prompt: str, config: GenerationConfig = None) -> Generator[str, None, None]:
        """Generate text as a stream using OpenAI API"""
        if config is None:
            config = GenerationConfig()
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop_sequences,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"Error streaming text with OpenAI: {e}")
            yield f"Error: {str(e)}"
    
    @property
    def model_name(self) -> str:
        return self.model


class LocalLLMProvider(LLMProvider):
    """Local LLM provider using ctransformers"""
    
    def __init__(self, model_path: str, model_type: str = "llama", config: Dict[str, Any] = None):
        self.model_path = model_path
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize local model"""
        try:
            from ctransformers import AutoModelForCausalLM
            
            default_config = {
                "max_new_tokens": 1000,
                "temperature": 0.7,
                "repetition_penalty": 1.1,
                "context_length": 4096
            }
            default_config.update(self.config)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_type=self.model_type,
                **default_config
            )
            
            logger.info(f"Local LLM initialized: {self.model_path}")
        
        except ImportError:
            logger.error("ctransformers not installed. Please install with: pip install ctransformers")
            raise
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            raise
    
    def generate(self, prompt: str, config: GenerationConfig = None) -> GenerationResult:
        """Generate text using local model"""
        if config is None:
            config = GenerationConfig()
        
        try:
            # Prepare generation parameters
            gen_params = {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "repetition_penalty": 1.1,
                "stop": config.stop_sequences or []
            }
            
            # Generate text
            output = self.model(prompt, **gen_params)
            
            return GenerationResult(
                text=output,
                usage={
                    "prompt_tokens": len(prompt.split()),  # Rough estimate
                    "completion_tokens": len(output.split()),
                    "total_tokens": len(prompt.split()) + len(output.split())
                },
                model=self.model_path,
                finish_reason="stop"
            )
        
        except Exception as e:
            logger.error(f"Error generating text with local model: {e}")
            return GenerationResult(
                text=f"Error: {str(e)}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.model_path,
                finish_reason="error"
            )
    
    def generate_stream(self, prompt: str, config: GenerationConfig = None) -> Generator[str, None, None]:
        """Generate text as a stream using local model"""
        if config is None:
            config = GenerationConfig()
        
        try:
            gen_params = {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "repetition_penalty": 1.1,
                "stop": config.stop_sequences or []
            }
            
            # Generate text token by token
            for token in self.model(prompt, stream=True, **gen_params):
                yield token
        
        except Exception as e:
            logger.error(f"Error streaming text with local model: {e}")
            yield f"Error: {str(e)}"
    
    @property
    def model_name(self) -> str:
        return f"local:{self.model_path}"


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Claude client"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"Claude provider initialized with model: {self.model}")
        except ImportError:
            logger.error("Anthropic library not installed. Please install with: pip install anthropic")
            raise
    
    def generate(self, prompt: str, config: GenerationConfig = None) -> GenerationResult:
        """Generate text using Claude API"""
        if config is None:
            config = GenerationConfig()
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return GenerationResult(
                text=response.content[0].text,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                model=response.model,
                finish_reason=response.stop_reason
            )
        
        except Exception as e:
            logger.error(f"Error generating text with Claude: {e}")
            return GenerationResult(
                text=f"Error: {str(e)}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.model,
                finish_reason="error"
            )
    
    def generate_stream(self, prompt: str, config: GenerationConfig = None) -> Generator[str, None, None]:
        """Generate text as a stream using Claude API"""
        if config is None:
            config = GenerationConfig()
        
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text
        
        except Exception as e:
            logger.error(f"Error streaming text with Claude: {e}")
            yield f"Error: {str(e)}"
    
    @property
    def model_name(self) -> str:
        return self.model


def create_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
    """Factory function to create LLM provider instances"""
    
    if provider_type.lower() == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type.lower() == "local":
        return LocalLLMProvider(**kwargs)
    elif provider_type.lower() == "claude":
        return ClaudeProvider(**kwargs)
    elif provider_type.lower() == "huggingface_local":
        return HuggingFaceLocalLLMProvider(**kwargs)
    elif provider_type.lower() == "groq":
        return GroqLLMProvider(**kwargs)
    elif provider_type.lower() == "ollama":
        return OllamaLLMProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_type}")


# Free LLM Provider Classes
class HuggingFaceLocalLLMProvider(LLMProvider):
    """Hugging Face local model provider"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self._model_name = model_name
        self.provider = None
        
    def _initialize(self):
        if self.provider is None:
            from .huggingface_local_provider import HuggingFaceLocalProvider
            self.provider = HuggingFaceLocalProvider(self._model_name)
            return self.provider.initialize()
        return True
    
    def generate(self, prompt: str, config: GenerationConfig = None) -> GenerationResult:
        if not self._initialize():
            return GenerationResult(
                text="Error: Failed to initialize Hugging Face model",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self._model_name,
                finish_reason="error"
            )
        
        response = self.provider.generate(prompt, 
            temperature=config.temperature if config else 0.8,
            max_tokens=config.max_tokens if config else 30  # Short responses for small models
        )
        
        return GenerationResult(
            text=response,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(response.split()), "total_tokens": len(prompt.split()) + len(response.split())},
            model=self._model_name,
            finish_reason="stop"
        )
    
    def generate_stream(self, prompt: str, config: GenerationConfig = None):
        response = self.generate(prompt, config)
        yield response.text
    
    @property 
    def model_name(self) -> str:
        return self._model_name


class GroqLLMProvider(LLMProvider):
    """Groq API provider"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self._model = model
        self.provider = None
        
    def _initialize(self):
        if self.provider is None:
            from .groq_provider import GroqProvider
            self.provider = GroqProvider(self.api_key, self._model)
            return self.provider.initialize()
        return True
    
    def generate(self, prompt: str, config: GenerationConfig = None) -> GenerationResult:
        if not self._initialize():
            return GenerationResult(
                text="Error: Failed to initialize Groq provider",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self._model,
                finish_reason="error"
            )
        
        response = self.provider.generate(prompt,
            temperature=config.temperature if config else 0.7,
            max_tokens=config.max_tokens if config else 1000
        )
        
        return GenerationResult(
            text=response,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(response.split()), "total_tokens": len(prompt.split()) + len(response.split())},
            model=self._model,
            finish_reason="stop"
        )
    
    def generate_stream(self, prompt: str, config: GenerationConfig = None):
        response = self.generate(prompt, config)
        yield response.text
    
    @property
    def model_name(self) -> str:
        return self._model


class OllamaLLMProvider(LLMProvider):
    """Ollama local provider"""
    
    def __init__(self, model: str = "llama3.1:8b"):
        self._model = model
        self.provider = None
        
    def _initialize(self):
        if self.provider is None:
            from .ollama_provider import OllamaProvider
            self.provider = OllamaProvider(self._model)
            return self.provider.initialize()
        return True
    
    def generate(self, prompt: str, config: GenerationConfig = None) -> GenerationResult:
        if not self._initialize():
            return GenerationResult(
                text="Error: Failed to initialize Ollama. Please install Ollama and run: ollama pull " + self._model,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self._model,
                finish_reason="error"
            )
        
        response = self.provider.generate(prompt,
            temperature=config.temperature if config else 0.7,
            max_tokens=config.max_tokens if config else 1000
        )
        
        return GenerationResult(
            text=response,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(response.split()), "total_tokens": len(prompt.split()) + len(response.split())},
            model=self._model,
            finish_reason="stop"
        )
    
    def generate_stream(self, prompt: str, config: GenerationConfig = None):
        response = self.generate(prompt, config)
        yield response.text
    
    @property
    def model_name(self) -> str:
        return self._model


class PromptTemplate:
    """Template for RAG prompts with inline citations"""
    
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables"""
        return self.template.format(**kwargs)


# Default RAG prompt templates
DEFAULT_RAG_PROMPT = """
Information: {context}

Question: {question}

Answer: 
"""

HEALTHCARE_RAG_PROMPT = """
Medical Information: {context}

Question: {question}

Answer: 
"""

LEGAL_RAG_PROMPT = """
You are a legal research assistant providing information based on legal documents and case law.

Legal Context:
{context}

Legal Question: {question}

Instructions:
1. Base your response entirely on the legal context provided above.
2. Include precise citations using [Source X] format for all legal references.
3. Use proper legal terminology and cite specific sections, cases, or statutes when mentioned.
4. If the context is insufficient to answer the question comprehensively, indicate this clearly.
5. Clarify that this is informational only and not legal advice.
6. Recommend consulting qualified legal counsel for specific legal matters.

Legal Analysis:
"""

FINANCIAL_RAG_PROMPT = """
You are a financial information assistant providing analysis based on financial documents and market data.

Financial Context:
{context}

Financial Question: {question}

Instructions:
1. Provide financial information based exclusively on the context above.
2. Include citations using [Source X] format for all financial data and analyses.
3. Use appropriate financial terminology and metrics.
4. If the context lacks sufficient data for a complete analysis, state this clearly.
5. Include appropriate disclaimers about market risks and volatility.
6. Recommend consulting financial advisors for investment decisions.

Financial Analysis:
"""

# Domain-specific prompt templates
DOMAIN_PROMPTS = {
    "general": DEFAULT_RAG_PROMPT,
    "healthcare": HEALTHCARE_RAG_PROMPT,
    "legal": LEGAL_RAG_PROMPT,
    "financial": FINANCIAL_RAG_PROMPT
}
