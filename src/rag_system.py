"""Main RAG Expert System class that orchestrates all components"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import uuid
import re
from datetime import datetime
from loguru import logger
import os

from .config import settings
from .chunking import AdaptiveChunker, Chunk
from .embeddings import create_embedding_provider, EmbeddingProvider
from .vector_stores import create_vector_store, VectorStore, Document
from .hybrid_retrieval import HybridRetriever, SearchResult
from .llm_providers import create_llm_provider, LLMProvider, GenerationConfig, DOMAIN_PROMPTS
from .document_processor import DocumentProcessingPipeline, ProcessedDocument
from .source_formatter import format_sources, format_sources_as_markdown


@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    model_used: str
    timestamp: datetime
    retrieval_method: str
    processing_time: float
    session_id: str

@dataclass
class AuditLog:
    """Audit log entry"""
    session_id: str
    query: str
    response: str
    sources_used: List[str]
    timestamp: datetime
    model_used: str
    retrieval_method: str
    processing_time: float
    user_feedback: Optional[str] = None

class RAGExpertSystem:
    """Main RAG Expert System class"""
    
    def __init__(self, domain: str = "general", session_id: str = None):
        """Initialize RAG Expert System
        
        Args:
            domain: Domain for specialized prompts (general, healthcare, legal, financial)
            session_id: Optional session ID for tracking
        """
        self.domain = domain
        self.session_id = session_id or str(uuid.uuid4())
        
        # Core components
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.vector_store: Optional[VectorStore] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.llm_provider: Optional[LLMProvider] = None
        self.chunker = AdaptiveChunker()
        self.document_processor = DocumentProcessingPipeline()
        
        # State
        self.is_initialized = False
        self.indexed_documents = []
        self.audit_logs = []
        
        logger.info(f"RAG Expert System created for domain: {domain}, session: {self.session_id}")
    
    def initialize(self) -> bool:
        """Initialize all components based on configuration"""
        try:
            self._initialize_embeddings()
            self._initialize_vector_store()
            self._initialize_llm()
            
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider,
                alpha=0.7
            )
            
            # Initialize hybrid retriever with existing documents if any
            try:
                all_docs = self.vector_store.get_all_documents()
                if all_docs:
                    logger.info(f"Found {len(all_docs)} existing documents, initializing hybrid retriever")
                    keyword_docs = []
                    for doc in all_docs:
                        keyword_docs.append({
                            'content': doc.content,
                            'metadata': doc.metadata or {},
                            'doc_id': doc.doc_id
                        })
                    if keyword_docs:
                        self.hybrid_retriever.index_documents(keyword_docs)
                        self.hybrid_retriever.documents_indexed = True
                        logger.info("Hybrid retriever successfully initialized with existing documents")
                    else:
                        logger.debug("No document content available for hybrid retriever indexing")
            except Exception as e:
                logger.warning(f"Could not initialize hybrid retriever with existing documents: {e}")
                # Create an empty index to avoid further warnings
                self.hybrid_retriever.keyword_searcher.add_documents([{
                    'content': 'Empty placeholder document', 
                    'metadata': {}, 
                    'doc_id': 'placeholder'
                }])
                self.hybrid_retriever.documents_indexed = True
            
            self.is_initialized = True
            logger.info("RAG Expert System initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def _initialize_embeddings(self):
        """Initialize embedding provider"""
        embedding_type = settings.embedding_model.lower()
        
        if embedding_type == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not provided")
            self.embedding_provider = create_embedding_provider(
                "openai",
                api_key=settings.openai_api_key,
                model="text-embedding-ada-002"
            )
        elif embedding_type == "sentence-transformers" or embedding_type == "huggingface":
            self.embedding_provider = create_embedding_provider(
                "huggingface",
                model_name="all-MiniLM-L6-v2"
            )
        elif embedding_type == "cohere":
            if not settings.cohere_api_key:
                raise ValueError("Cohere API key not provided")
            self.embedding_provider = create_embedding_provider(
                "cohere",
                api_key=settings.cohere_api_key
            )
        elif embedding_type == "local":
            self.embedding_provider = create_embedding_provider(
                "local",
                model_path=settings.local_model_path
            )
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_type}")
    
    def _initialize_vector_store(self):
        """Initialize vector store"""
        store_type = settings.vector_store_type.lower()
        
        if store_type == "chroma":
            self.vector_store = create_vector_store(
                "chroma",
                collection_name=f"rag_{self.domain}",
                persist_directory=f"./chroma_db_{self.domain}"
            )
        elif store_type == "pinecone":
            if not settings.pinecone_api_key or not settings.pinecone_environment:
                raise ValueError("Pinecone credentials not provided")
            self.vector_store = create_vector_store(
                "pinecone",
                index_name=f"rag-{self.domain}",
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment,
                dimension=self.embedding_provider.dimension
            )
        elif store_type == "weaviate":
            self.vector_store = create_vector_store(
                "weaviate",
                url=settings.weaviate_url,
                api_key=settings.weaviate_api_key,
                class_name=f"RAG{self.domain.capitalize()}"
            )
        else:
            raise ValueError(f"Unsupported vector store: {store_type}")
    
    def _initialize_llm(self):
        """Initialize LLM provider"""
        if settings.use_local_model or settings.llm_model in ["huggingface_local", "ollama"]:
            if settings.llm_model == "huggingface_local":
                model_name = getattr(settings, 'local_model_name', 'microsoft/DialoGPT-medium')
                self.llm_provider = create_llm_provider(
                    "huggingface_local",
                    model_name=model_name
                )
            elif settings.llm_model == "ollama":
                model_name = getattr(settings, 'ollama_model', 'llama3.1:8b')
                self.llm_provider = create_llm_provider(
                    "ollama", 
                    model=model_name
                )
            else:
                self.llm_provider = create_llm_provider(
                    "local",
                    model_path=settings.local_model_path,
                    model_type="llama"
                )
        elif settings.llm_model == "groq":
            groq_api_key = getattr(settings, 'groq_api_key', None)
            if not groq_api_key:
                raise ValueError("Groq API key not provided. Set GROQ_API_KEY in .env file")
            groq_model = getattr(settings, 'groq_model', 'llama-3.1-8b-instant')
            self.llm_provider = create_llm_provider(
                "groq",
                api_key=groq_api_key,
                model=groq_model
            )
        elif settings.llm_model.startswith("gpt"):
            if not settings.openai_api_key or settings.openai_api_key == "not_needed_for_local":
                raise ValueError("OpenAI API key not provided")
            self.llm_provider = create_llm_provider(
                "openai",
                api_key=settings.openai_api_key,
                model=settings.llm_model
            )
        elif settings.llm_model.startswith("claude"):
            raise NotImplementedError("Claude provider not fully implemented")
        else:
            raise ValueError(f"Unsupported LLM model: {settings.llm_model}")
    
    def add_documents(self, file_paths: List[str], metadata: Dict[str, Any] = None) -> bool:
        """Add documents to the knowledge base"""
        if not self.is_initialized:
            logger.error("System not initialized. Call initialize() first.")
            return False
        
        try:
            all_chunks = []
            
            for file_path in file_paths:
                processed_doc = self.document_processor.process_file(file_path, metadata)
                if not processed_doc:
                    logger.warning(f"Failed to process document: {file_path}")
                    continue
                
                # Create base metadata dict with safe fallbacks
                doc_metadata = {
                    "source_file": processed_doc.source_path,
                    "doc_type": processed_doc.doc_type
                }
                
                # Safely merge additional metadata from processed_doc
                if hasattr(processed_doc, 'metadata'):
                    if processed_doc.metadata is None:
                        # Nothing to merge if metadata is None
                        pass
                    elif isinstance(processed_doc.metadata, dict):
                        # Safe update with dict comprehension to avoid modifying original
                        for k, v in processed_doc.metadata.items():
                            doc_metadata[k] = v
                    elif isinstance(processed_doc.metadata, list):
                        # If metadata is a list, store it as a special key
                        doc_metadata["metadata_list"] = processed_doc.metadata
                    else:
                        # For any other type, store as string in a special key
                        doc_metadata["original_metadata"] = str(processed_doc.metadata)
                
                chunks = self.chunker.chunk(
                    processed_doc.content,
                    doc_metadata
                )
                
                all_chunks.extend(chunks)
                self.indexed_documents.append(processed_doc.source_path)
            
            if not all_chunks:
                logger.warning("No chunks created from documents")
                return False
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding_provider.embed_texts(chunk_texts)
            
            # Create documents for vector store
            documents = []
            keyword_docs = []
            
            for chunk, embedding in zip(all_chunks, embeddings):
                # Ensure metadata is a dictionary with safe handling
                if hasattr(chunk, 'metadata'):
                    if chunk.metadata is None:
                        # Use empty dict if metadata is None
                        safe_metadata = {}
                    elif isinstance(chunk.metadata, dict):
                        # Make a clean copy of the metadata dict
                        safe_metadata = {}
                        for k, v in chunk.metadata.items():
                            safe_metadata[k] = v
                    else:
                        # For non-dict metadata, store it in a special key
                        safe_metadata = {"original_metadata": str(chunk.metadata)}
                        logger.debug(f"Converted non-dict chunk metadata to dict in add_documents: {type(chunk.metadata).__name__}")
                else:
                    # Default to empty dict if no metadata attribute
                    safe_metadata = {}
                
                doc = Document(
                    content=chunk.content,
                    embedding=embedding,
                    metadata=safe_metadata,
                    doc_id=chunk.chunk_id
                )
                documents.append(doc)
                
                keyword_docs.append({
                    'content': chunk.content,
                    'metadata': safe_metadata,
                    'doc_id': chunk.chunk_id
                })
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents(documents)
            
            # Index for hybrid retrieval
            if hasattr(self.hybrid_retriever, 'index_documents') and keyword_docs:
                try:
                    # If this is the first time adding documents, initialize the index
                    if not self.hybrid_retriever.documents_indexed:
                        # Get all documents from vector store to build a complete index
                        try:
                            all_docs = self.vector_store.get_all_documents()
                            if all_docs:
                                all_keyword_docs = []
                                for doc in all_docs:
                                    all_keyword_docs.append({
                                        'content': doc.content,
                                        'metadata': doc.metadata or {},
                                        'doc_id': doc.doc_id
                                    })
                                if all_keyword_docs:
                                    self.hybrid_retriever.index_documents(all_keyword_docs)
                                    self.hybrid_retriever.documents_indexed = True
                                    logger.info(f"Initialized hybrid retriever with all {len(all_keyword_docs)} existing documents")
                                    # Skip adding the current documents again since we just indexed everything
                                    return True
                        except Exception as e:
                            logger.warning(f"Failed to get all documents for hybrid index initialization: {e}")
                            # Continue with just adding the new documents
                    
                    # Index just the new documents
                    self.hybrid_retriever.index_documents(keyword_docs)
                    self.hybrid_retriever.documents_indexed = True
                    logger.info(f"Added {len(keyword_docs)} documents to hybrid index")
                except Exception as e:
                    logger.warning(f"Error indexing documents for hybrid search: {e}")
                    # We'll still return True even if hybrid indexing fails, as the documents are in the vector store
            
            logger.info(f"Successfully added {len(all_chunks)} chunks from {len(file_paths)} documents")
                    
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def add_directory(self, directory_path: str, recursive: bool = True, 
                     metadata: Dict[str, Any] = None) -> bool:
        """Add all supported documents from a directory"""
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return False
        
        # Get all supported files
        file_paths = []
        supported_formats = self.document_processor.get_supported_formats()
        
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if any(file.lower().endswith(fmt) for fmt in supported_formats):
                        file_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if (os.path.isfile(file_path) and 
                    any(file.lower().endswith(fmt) for fmt in supported_formats)):
                    file_paths.append(file_path)
        
        if not file_paths:
            logger.warning(f"No supported documents found in: {directory_path}")
            return False
        
        return self.add_documents(file_paths, metadata)
    
    def query(self, question: str, max_chunks: int = None, 
              generation_config: GenerationConfig = None,
              retrieval_method: str = "hybrid",
              custom_search_params: Dict[str, Any] = None) -> RAGResponse:
        """Query the RAG system"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        start_time = datetime.now()
        max_chunks = max_chunks or settings.max_chunks_per_query
        
        # Default relevance threshold
        relevance_threshold = 0.0
        document_filter = None
        
        # Process custom search parameters
        if custom_search_params:
            logger.info(f"Processing custom search parameters: {custom_search_params}")
            
            # Extract relevance threshold if provided
            relevance_threshold = custom_search_params.get("relevance_threshold", 0.0)
            
            # Extract document filter if provided
            if "filter" in custom_search_params:
                document_filter = custom_search_params["filter"]
                logger.info(f"Using document filter: {document_filter}")
            
        try:
            # Retrieve relevant documents
            # Pass recency_boost parameter if specified in custom_search_params
            recency_boost = custom_search_params.get("recency_boost", False) if custom_search_params else False
            
            if retrieval_method == "hybrid":
                search_results = self.hybrid_retriever.search(
                    question, k=max_chunks*2, recency_boost=recency_boost
                )  # Get more results than needed for filtering
            elif retrieval_method == "vector":
                search_results = self.hybrid_retriever.search(
                    question, k=max_chunks*2, use_vector=True, use_keyword=False, recency_boost=recency_boost
                )
            elif retrieval_method == "keyword":
                search_results = self.hybrid_retriever.search(
                    question, k=max_chunks*2, use_vector=False, use_keyword=True, recency_boost=recency_boost
                )
            else:
                raise ValueError(f"Invalid retrieval method: {retrieval_method}")
            
            # Filter results based on relevance threshold and metadata filters
            filtered_results = []
            for result in search_results:
                # Skip documents below relevance threshold
                if result.score < relevance_threshold:
                    logger.debug(f"Skipping document with low relevance score: {result.score}")
                    continue
                
                # Apply metadata filters if provided
                if document_filter and result.metadata:
                    # Check if the document meets all filter criteria
                    matches_filter = all(
                        result.metadata.get(key) == value 
                        for key, value in document_filter.items()
                    )
                    if not matches_filter:
                        logger.debug(f"Skipping document that doesn't match filter: {document_filter}")
                        continue
                
                filtered_results.append(result)
                
                # Break once we have enough results
                if len(filtered_results) >= max_chunks:
                    break
            
            search_results = filtered_results
            
            if not search_results:
                return RAGResponse(
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    query=question,
                    model_used=self.llm_provider.model_name,
                    timestamp=datetime.now(),
                    retrieval_method=retrieval_method,
                    processing_time=0.0,
                    session_id=self.session_id
                )
            
            # Prepare context with inline citations
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results, 1):
                # Extract and enhance page information for PDFs
                page_num = None
                if result.metadata and result.content:
                    page_match = re.search(r"--- Page (\d+) ---", result.content[:50])
                    if page_match:
                        page_num = page_match.group(1)
                        result.metadata["page"] = int(page_num)
                
                # Create enhanced source info
                source_info = {
                    "source_id": i,
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata,
                    "doc_id": result.doc_id,
                    "source": result.metadata.get('source_file', 'Unknown')
                }
                sources.append(source_info)
                
                # Add source reference (truncate content to stay within model limits)
                source_file = result.metadata.get('source_file', 'Unknown')
                truncated_content = result.content[:300] + "..." if len(result.content) > 300 else result.content
                
                # Add page information if available
                source_citation = f"[Source {i}] ({source_file})"
                if page_num:
                    source_citation += f" Page {page_num}"
                
                context_parts.append(f"{source_citation}: {truncated_content}")
            
            context = "\n\n".join(context_parts)
            
            # Generate prompt
            prompt_template = DOMAIN_PROMPTS.get(self.domain, DOMAIN_PROMPTS["general"])
            prompt = prompt_template.format(context=context, question=question)
            
            # Generate response
            if generation_config is None:
                generation_config = GenerationConfig(temperature=0.3, max_tokens=1000)
            
            llm_result = self.llm_provider.generate(prompt, generation_config)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = RAGResponse(
                answer=llm_result.text,
                sources=sources,
                query=question,
                model_used=self.llm_provider.model_name,
                timestamp=datetime.now(),
                retrieval_method=retrieval_method,
                processing_time=processing_time,
                session_id=self.session_id
            )
            
            # Log for audit trail
            if settings.enable_audit_logs:
                self._log_interaction(question, llm_result.text, sources, 
                                    retrieval_method, processing_time)
            
            return response
        
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return RAGResponse(
                answer=f"An error occurred while processing your query: {str(e)}",
                sources=[],
                query=question,
                model_used=self.llm_provider.model_name if self.llm_provider else "unknown",
                timestamp=datetime.now(),
                retrieval_method=retrieval_method,
                processing_time=(datetime.now() - start_time).total_seconds(),
                session_id=self.session_id
            )
    
    def _log_interaction(self, query: str, response: str, sources: List[Dict[str, Any]],
                        retrieval_method: str, processing_time: float):
        """Log interaction for audit trail"""
        audit_entry = AuditLog(
            session_id=self.session_id,
            query=query if not settings.anonymize_queries else "***ANONYMIZED***",
            response=response,
            sources_used=[s.get('doc_id', '') for s in sources],
            timestamp=datetime.now(),
            model_used=self.llm_provider.model_name,
            retrieval_method=retrieval_method,
            processing_time=processing_time
        )
        
        self.audit_logs.append(audit_entry)
        
        # Save to file
        if settings.log_file:
            try:
                os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
                with open(settings.log_file.replace('.log', '_audit.jsonl'), 'a') as f:
                    f.write(json.dumps({
                        'session_id': audit_entry.session_id,
                        'query': audit_entry.query,
                        'response': audit_entry.response,
                        'sources_used': audit_entry.sources_used,
                        'timestamp': audit_entry.timestamp.isoformat(),
                        'model_used': audit_entry.model_used,
                        'retrieval_method': audit_entry.retrieval_method,
                        'processing_time': audit_entry.processing_time
                    }) + '\n')
            except Exception as e:
                logger.error(f"Failed to save audit log: {e}")
    
    def explain_retrieval(self, question: str, max_chunks: int = None, 
                     custom_search_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Explain how retrieval works for a given question"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        max_chunks = max_chunks or settings.max_chunks_per_query
        
        # Default relevance threshold
        relevance_threshold = 0.0
        document_filter = None
        
        # Process custom search parameters
        if custom_search_params:
            logger.info(f"Processing custom search parameters in explain_retrieval: {custom_search_params}")
            
            # Extract relevance threshold if provided
            relevance_threshold = custom_search_params.get("relevance_threshold", 0.0)
            
            # Extract document filter if provided
            if "filter" in custom_search_params:
                document_filter = custom_search_params["filter"]
                logger.info(f"Using document filter: {document_filter}")
        
        # Get explanation with raw results
        explanation = self.hybrid_retriever.explain_search(question, k=max_chunks*2)
        
        # Apply filters to results
        for result_type in ["vector_results", "keyword_results", "hybrid_results"]:
            filtered_results = []
            for result in explanation[result_type]:
                # Skip documents below relevance threshold
                if result.score < relevance_threshold:
                    continue
                
                # Apply metadata filters if provided
                if document_filter and result.metadata:
                    # Check if the document meets all filter criteria
                    matches_filter = all(
                        result.metadata.get(key) == value 
                        for key, value in document_filter.items()
                    )
                    if not matches_filter:
                        continue
                
                filtered_results.append(result)
                
                # Break once we have enough results
                if len(filtered_results) >= max_chunks:
                    break
                    
            explanation[result_type] = filtered_results[:max_chunks]
            
        # Add filtering info to explanation
        explanation["filters_applied"] = {
            "relevance_threshold": relevance_threshold,
            "document_filter": document_filter,
            "max_chunks": max_chunks
        }
        
        return explanation
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "domain": self.domain,
            "session_id": self.session_id,
            "is_initialized": self.is_initialized,
            "indexed_documents": len(self.indexed_documents),
            "total_queries": len(self.audit_logs),
            "embedding_model": self.embedding_provider.model_name if self.embedding_provider else None,
            "vector_store_type": settings.vector_store_type,
            "llm_model": self.llm_provider.model_name if self.llm_provider else None,
            "supported_formats": self.document_processor.get_supported_formats()
        }
    
    def export_audit_logs(self, file_path: str = None) -> str:
        """Export audit logs to file"""
        if not file_path:
            file_path = f"audit_logs_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        logs_data = []
        for log in self.audit_logs:
            logs_data.append({
                'session_id': log.session_id,
                'query': log.query,
                'response': log.response,
                'sources_used': log.sources_used,
                'timestamp': log.timestamp.isoformat(),
                'model_used': log.model_used,
                'retrieval_method': log.retrieval_method,
                'processing_time': log.processing_time,
                'user_feedback': log.user_feedback
            })
        
        with open(file_path, 'w') as f:
            json.dump(logs_data, f, indent=2)
        
        logger.info(f"Audit logs exported to: {file_path}")
        return file_path
    
    def query(self, query_text: str) -> RAGResponse:
        """Query the RAG system with a natural language question"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        start_time = datetime.now()
        
        try:
            # Retrieve relevant documents
            search_results = self.hybrid_retriever.search(query_text, k=settings.max_chunks_per_query)
            
            if not search_results:
                return RAGResponse(
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    query=query_text,
                    model_used=self.llm_provider.model_name,
                    timestamp=datetime.now(),
                    retrieval_method="hybrid",
                    processing_time=0.0,
                    session_id=self.session_id
                )
            
            # Prepare context with inline citations
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results, 1):
                # Extract and enhance page information for PDFs
                page_num = None
                if result.metadata and result.content:
                    page_match = re.search(r"--- Page (\d+) ---", result.content[:50])
                    if page_match:
                        page_num = page_match.group(1)
                        result.metadata["page"] = int(page_num)
                
                # Create enhanced source info
                source_info = {
                    "source_id": i,
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata,
                    "doc_id": result.doc_id,
                    "source": result.metadata.get('source_file', 'Unknown')
                }
                sources.append(source_info)
                
                # Add source reference (truncate content to stay within model limits)
                source_file = result.metadata.get('source_file', 'Unknown')
                truncated_content = result.content[:300] + "..." if len(result.content) > 300 else result.content
                
                # Add page information if available
                source_citation = f"[Source {i}] ({source_file})"
                if page_num:
                    source_citation += f" Page {page_num}"
                
                context_parts.append(f"{source_citation}: {truncated_content}")
            
            context = "\n\n".join(context_parts)
            
            # Generate prompt
            prompt_template = DOMAIN_PROMPTS.get(self.domain, DOMAIN_PROMPTS["general"])
            prompt = prompt_template.format(context=context, question=query_text)
            
            # Generate response
            llm_result = self.llm_provider.generate(prompt, GenerationConfig(temperature=0.3, max_tokens=1000))
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = RAGResponse(
                answer=llm_result.text,
                sources=sources,
                query=query_text,
                model_used=self.llm_provider.model_name,
                timestamp=datetime.now(),
                retrieval_method="hybrid",
                processing_time=processing_time,
                session_id=self.session_id
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return RAGResponse(
                answer=f"An error occurred while processing your query: {str(e)}",
                sources=[],
                query=query_text,
                model_used=self.llm_provider.model_name if self.llm_provider else "unknown",
                timestamp=datetime.now(),
                retrieval_method="hybrid",
                processing_time=(datetime.now() - start_time).total_seconds(),
                session_id=self.session_id
            )
    
    def get_all_document_ids(self) -> List[str]:
        """Get all document IDs in the vector store"""
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
            
        try:
            # Try to use the get_all_documents method if available
            if hasattr(self.vector_store, 'get_all_documents'):
                docs = self.vector_store.get_all_documents()
                return [doc.doc_id for doc in docs]
            # Fallback for ChromaDB
            elif hasattr(self.vector_store, 'collection'):
                all_docs = self.vector_store.collection.get(
                    include=["ids"],
                    limit=10000  # Set a reasonable limit
                )
                if all_docs and 'ids' in all_docs:
                    return all_docs['ids']
        except Exception as e:
            logger.error(f"Error getting document IDs: {e}")
        
        return []
