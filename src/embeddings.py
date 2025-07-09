"""Embeddings module for different embedding providers"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension"""
        pass

class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings provider"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI embeddings initialized with model: {self.model}")
        except ImportError:
            logger.error("OpenAI library not installed. Please install with: pip install openai")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.dimension)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                # Add zero embeddings for failed batch
                embeddings.extend([np.zeros(self.dimension) for _ in batch])
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if self.model == "text-embedding-ada-002":
            return 1536
        elif self.model == "text-embedding-3-small":
            return 1536
        elif self.model == "text-embedding-3-large":
            return 3072
        else:
            return 1536  # Default


class HuggingFaceEmbeddings(EmbeddingProvider):
    """Hugging Face sentence transformers embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"HuggingFace embeddings initialized with model: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed. Please install with: pip install sentence-transformers")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.dimension)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return [emb for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [np.zeros(self.dimension) for _ in texts]
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        else:
            # Common dimensions for popular models
            model_dimensions = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "all-distilroberta-v1": 768,
                "paraphrase-MiniLM-L6-v2": 384,
            }
            return model_dimensions.get(self.model_name, 384)


class CohereEmbeddings(EmbeddingProvider):
    """Cohere embeddings provider"""
    
    def __init__(self, api_key: str, model: str = "embed-english-v2.0"):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Cohere client"""
        try:
            import cohere
            self.client = cohere.Client(api_key=self.api_key)
            logger.info(f"Cohere embeddings initialized with model: {self.model}")
        except ImportError:
            logger.error("Cohere library not installed. Please install with: pip install cohere")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model
            )
            return np.array(response.embeddings[0])
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.dimension)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        try:
            # Process in batches
            batch_size = 96  # Cohere's batch limit
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.client.embed(
                    texts=batch,
                    model=self.model
                )
                
                batch_embeddings = [np.array(emb) for emb in response.embeddings]
                embeddings.extend(batch_embeddings)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [np.zeros(self.dimension) for _ in texts]
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if "large" in self.model:
            return 4096
        else:
            return 1024  # Default for most Cohere models


class LocalEmbeddings(EmbeddingProvider):
    """Local embeddings using downloaded models"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize local model"""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            
            if os.path.exists(self.model_path):
                self.model = SentenceTransformer(self.model_path, device=self.device)
                logger.info(f"Local embeddings initialized from: {self.model_path}")
            else:
                logger.error(f"Model path not found: {self.model_path}")
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
        except ImportError:
            logger.error("sentence-transformers not installed. Please install with: pip install sentence-transformers")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.dimension)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return [emb for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [np.zeros(self.dimension) for _ in texts]
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        else:
            return 384  # Default dimension


def create_embedding_provider(provider_type: str, **kwargs) -> EmbeddingProvider:
    """Factory function to create embedding provider instances"""
    
    if provider_type.lower() == "openai":
        return OpenAIEmbeddings(**kwargs)
    elif provider_type.lower() == "huggingface" or provider_type.lower() == "sentence-transformers":
        return HuggingFaceEmbeddings(**kwargs)
    elif provider_type.lower() == "cohere":
        return CohereEmbeddings(**kwargs)
    elif provider_type.lower() == "local":
        return LocalEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider_type}")
