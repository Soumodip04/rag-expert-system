"""Vector store implementations for different backends"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from loguru import logger
import json
import os

@dataclass
class Document:
    """Represents a document with embedding"""
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    doc_id: str = ""
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Ensure metadata is always a dictionary with ChromaDB-compatible values"""
        if self.metadata is None:
            self.metadata = {}
        elif not isinstance(self.metadata, dict):
            # Handle non-dict metadata by storing it in a special key
            original = self.metadata
            self.metadata = {"_document_original_metadata": str(original)}
            logger.debug(f"Converted non-dict document metadata to dict: {type(original).__name__}")
        
        # Add timestamp if not already present (used for prioritizing new documents)
        if self.timestamp is None:
            import time
            self.timestamp = time.time()
            # Also add to metadata for retrieval
            self.metadata["timestamp"] = str(self.timestamp)
        
        # Convert metadata values to ChromaDB-compatible types
        self._ensure_chroma_compatible_metadata()
    
    def _ensure_chroma_compatible_metadata(self):
        """Ensure all metadata values are compatible with ChromaDB (str, int, float, bool, None)"""
        for key, value in list(self.metadata.items()):
            if isinstance(value, list):
                # Convert lists to comma-separated strings
                self.metadata[key] = ",".join(map(str, value)) if value else ""
            elif isinstance(value, dict):
                # Convert dicts to string representation
                self.metadata[key] = str(value)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                # Convert any other non-compatible type to string
                self.metadata[key] = str(value)
                logger.debug(f"Converted non-compatible metadata value type {type(value).__name__} to string")

class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from the store"""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID"""
        pass

class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
        except ImportError:
            logger.error("ChromaDB not installed. Please install with: pip install chromadb")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to ChromaDB"""
        if not documents:
            return []
        
        doc_ids = []
        embeddings = []
        contents = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            doc_id = doc.doc_id or f"doc_{i}_{hash(doc.content)}"
            doc_ids.append(doc_id)
            embeddings.append(doc.embedding.tolist() if doc.embedding is not None else None)
            contents.append(doc.content)
            
            # Ensure metadata is compatible with ChromaDB
            if not doc.metadata:
                metadatas.append({})
            else:
                # The Document class should have already sanitized metadata, but let's double-check
                doc._ensure_chroma_compatible_metadata()
                metadatas.append(doc.metadata)
        
        # Filter out documents without embeddings
        valid_docs = [(i, doc_id) for i, doc_id in enumerate(doc_ids) if embeddings[i] is not None]
        
        if valid_docs:
            valid_indices = [i for i, _ in valid_docs]
            valid_ids = [doc_ids[i] for i in valid_indices]
            valid_embeddings = [embeddings[i] for i in valid_indices]
            valid_contents = [contents[i] for i in valid_indices]
            valid_metadatas = [metadatas[i] for i in valid_indices]
            
            self.collection.add(
                ids=valid_ids,
                embeddings=valid_embeddings,
                documents=valid_contents,
                metadatas=valid_metadatas
            )
            
            logger.info(f"Added {len(valid_docs)} documents to ChromaDB")
            return valid_ids
        
        return []
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents in ChromaDB"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            documents_with_scores = []
            
            for i in range(len(results['ids'][0])):
                doc = Document(
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    doc_id=results['ids'][0][i]
                )
                score = 1.0 - results['distances'][0][i]  # Convert distance to similarity
                documents_with_scores.append((doc, score))
            
            return documents_with_scores
        
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from ChromaDB"""
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID from ChromaDB"""
        try:
            result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
            
            if result['ids'] and len(result['ids']) > 0:
                return Document(
                    content=result['documents'][0],
                    metadata=result['metadatas'][0],
                    doc_id=result['ids'][0]
                )
            return None
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
            
    def get_all_documents(self) -> List[Document]:
        """Get all documents from ChromaDB"""
        try:
            # Get all document IDs first - limit to a reasonable number
            result = self.collection.get(include=["documents", "metadatas", "embeddings"], limit=10000)
            
            documents = []
            if result['ids'] and len(result['ids']) > 0:
                for i, doc_id in enumerate(result['ids']):
                    doc = Document(
                        content=result['documents'][i],
                        metadata=result['metadatas'][i],
                        doc_id=doc_id
                    )
                    documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents from ChromaDB")
            return documents
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation"""
    
    def __init__(self, index_name: str, api_key: str, environment: str, dimension: int = 1536):
        self.index_name = index_name
        self.api_key = api_key
        self.environment = environment
        self.dimension = dimension
        self.index = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone client"""
        try:
            import pinecone
            
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine"
                )
            
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Pinecone initialized with index: {self.index_name}")
        
        except ImportError:
            logger.error("Pinecone client not installed. Please install with: pip install pinecone-client")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to Pinecone"""
        if not documents:
            return []
        
        vectors = []
        
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                continue
            
            doc_id = doc.doc_id or f"doc_{i}_{hash(doc.content)}"
            
            vector = {
                "id": doc_id,
                "values": doc.embedding.tolist(),
                "metadata": {
                    "content": doc.content,
                    **(doc.metadata or {})
                }
            }
            vectors.append(vector)
        
        if vectors:
            self.index.upsert(vectors=vectors)
            logger.info(f"Added {len(vectors)} documents to Pinecone")
            return [v["id"] for v in vectors]
        
        return []
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents in Pinecone"""
        try:
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=k,
                include_metadata=True
            )
            
            documents_with_scores = []
            
            for match in results['matches']:
                metadata = match['metadata']
                content = metadata.pop('content', '')
                
                doc = Document(
                    content=content,
                    metadata=metadata,
                    doc_id=match['id']
                )
                documents_with_scores.append((doc, match['score']))
            
            return documents_with_scores
        
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from Pinecone"""
        try:
            self.index.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID from Pinecone"""
        try:
            result = self.index.fetch(ids=[doc_id])
            
            if doc_id in result['vectors']:
                vector_data = result['vectors'][doc_id]
                metadata = vector_data['metadata']
                content = metadata.pop('content', '')
                
                return Document(
                    content=content,
                    metadata=metadata,
                    doc_id=doc_id
                )
            return None
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None


class WeaviateVectorStore(VectorStore):
    """Weaviate vector store implementation"""
    
    def __init__(self, url: str, api_key: Optional[str] = None, class_name: str = "Document"):
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Weaviate client"""
        try:
            import weaviate
            
            if self.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.api_key)
                self.client = weaviate.Client(url=self.url, auth_client_secret=auth_config)
            else:
                self.client = weaviate.Client(url=self.url)
            
            # Create schema if it doesn't exist
            self._create_schema()
            logger.info(f"Weaviate initialized with class: {self.class_name}")
        
        except ImportError:
            logger.error("Weaviate client not installed. Please install with: pip install weaviate-client")
            raise
    
    def _create_schema(self):
        """Create Weaviate schema"""
        schema = {
            "class": self.class_name,
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Document content"
                },
                {
                    "name": "metadata",
                    "dataType": ["text"],
                    "description": "Document metadata as JSON string"
                }
            ]
        }
        
        try:
            if not self.client.schema.exists(self.class_name):
                self.client.schema.create_class(schema)
        except Exception as e:
            logger.warning(f"Schema creation warning: {e}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to Weaviate"""
        if not documents:
            return []
        
        doc_ids = []
        
        with self.client.batch as batch:
            for i, doc in enumerate(documents):
                if doc.embedding is None:
                    continue
                
                doc_id = doc.doc_id or f"doc_{i}_{hash(doc.content)}"
                
                properties = {
                    "content": doc.content,
                    "metadata": json.dumps(doc.metadata or {})
                }
                
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    uuid=doc_id,
                    vector=doc.embedding.tolist()
                )
                
                doc_ids.append(doc_id)
        
        logger.info(f"Added {len(doc_ids)} documents to Weaviate")
        return doc_ids
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents in Weaviate"""
        try:
            result = (
                self.client.query
                .get(self.class_name, ["content", "metadata"])
                .with_near_vector({"vector": query_embedding.tolist()})
                .with_limit(k)
                .with_additional(["certainty", "id"])
                .do()
            )
            
            documents_with_scores = []
            
            if "data" in result and "Get" in result["data"]:
                for item in result["data"]["Get"][self.class_name]:
                    try:
                        metadata = json.loads(item["metadata"]) if item["metadata"] else {}
                    except:
                        metadata = {}
                    
                    doc = Document(
                        content=item["content"],
                        metadata=metadata,
                        doc_id=item["_additional"]["id"]
                    )
                    score = item["_additional"]["certainty"]
                    documents_with_scores.append((doc, score))
            
            return documents_with_scores
        
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from Weaviate"""
        try:
            for doc_id in doc_ids:
                self.client.data_object.delete(uuid=doc_id, class_name=self.class_name)
            
            logger.info(f"Deleted {len(doc_ids)} documents from Weaviate")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID from Weaviate"""
        try:
            result = self.client.data_object.get_by_id(
                uuid=doc_id,
                class_name=self.class_name
            )
            
            if result:
                properties = result["properties"]
                try:
                    metadata = json.loads(properties["metadata"]) if properties.get("metadata") else {}
                except:
                    metadata = {}
                
                return Document(
                    content=properties["content"],
                    metadata=metadata,
                    doc_id=doc_id
                )
            return None
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None


def create_vector_store(store_type: str, **kwargs) -> VectorStore:
    """Factory function to create vector store instances"""
    
    if store_type.lower() == "chroma":
        return ChromaVectorStore(**kwargs)
    elif store_type.lower() == "pinecone":
        return PineconeVectorStore(**kwargs)
    elif store_type.lower() == "weaviate":
        return WeaviateVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
