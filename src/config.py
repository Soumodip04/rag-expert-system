"""Configuration management for RAG Expert System"""
import os
from typing import Optional, Union
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    weaviate_url: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    
    # LLM Configuration
    temperature: Union[float, str] = Field(default=0.7, env="TEMPERATURE")
    max_tokens: Union[int, str] = Field(default=1000, env="MAX_TOKENS")
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Free API Keys
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    
    # Vector Store Configuration
    vector_store_type: str = Field(default="chroma", env="VECTOR_STORE_TYPE")
    embedding_model: str = Field(default="sentence-transformers", env="EMBEDDING_MODEL")
    llm_model: str = Field(default="huggingface_local", env="LLM_MODEL")
    
    # Local Model Support
    use_local_model: bool = Field(default=True, env="USE_LOCAL_MODEL")
    local_model_path: str = Field(default="models/", env="LOCAL_MODEL_PATH")
    local_model_name: str = Field(default="microsoft/DialoGPT-medium", env="LOCAL_MODEL_NAME")
    
    # Ollama Configuration
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    
    # Groq Configuration  
    groq_model: str = Field(default="llama-3.1-8b-instant", env="GROQ_MODEL")
    
    # Document Processing
    chunk_size: Union[int, str] = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: Union[int, str] = Field(default=200, env="CHUNK_OVERLAP")
    use_semantic_chunking: Union[bool, str] = Field(default=True, env="USE_SEMANTIC_CHUNKING")
    max_chunks_per_query: Union[int, str] = Field(default=5, env="MAX_CHUNKS_PER_QUERY")
    
    # Retrieval Settings
    retrieval_method: str = Field(default="hybrid", env="RETRIEVAL_METHOD")
    
    # Embedding Configuration
    embedding_provider: str = Field(default="openai", env="EMBEDDING_PROVIDER")
    embedding_dimension: Union[int, str] = Field(default=1536, env="EMBEDDING_DIMENSION")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/rag_system.log", env="LOG_FILE")
    
    # Privacy and Security
    enable_audit_logs: bool = Field(default=True, env="ENABLE_AUDIT_LOGS")
    anonymize_queries: bool = Field(default=False, env="ANONYMIZE_QUERIES")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields not defined in the model


# Global settings instance
settings = Settings()
