"""Hybrid retrieval system combining vector search and keyword search"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from loguru import logger
import re
from collections import Counter
import math

@dataclass
class SearchResult:
    """Search result with content and score"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str  # "vector", "keyword", or "hybrid"
    doc_id: str = ""

class KeywordSearcher:
    """BM25-based keyword search implementation"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_frequencies = {}
        self.avg_doc_length = 0
        self.doc_lengths = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the keyword index"""
        try:
            # Validate and clean documents before processing
            clean_documents = []
            for doc in documents:
                if not isinstance(doc, dict):
                    logger.warning(f"Skipping non-dictionary document: {type(doc)}")
                    continue
                    
                # Ensure metadata is a dictionary
                if 'metadata' in doc and not isinstance(doc['metadata'], dict):
                    doc['metadata'] = {}
                
                clean_documents.append(doc)
            
            self.documents = clean_documents
            self.doc_lengths = []
            word_doc_count = Counter()
            
            # Calculate document frequencies and lengths
            for doc in clean_documents:
                content = doc.get('content', '')
                if not content:
                    continue
                    
                words = self._tokenize(content)
                self.doc_lengths.append(len(words))
                
                # Count unique words in this document
                unique_words = set(words)
                for word in unique_words:
                    word_doc_count[word] += 1
            
            # Calculate average document length
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
            
            # Store document frequencies
            total_docs = len(clean_documents)
            self.doc_frequencies = {word: count for word, count in word_doc_count.items()}
            
            logger.info(f"Keyword search index built with {total_docs} documents")
        except Exception as e:
            logger.error(f"Error building keyword index: {e}")
            # Initialize with empty data to prevent errors
            self.documents = []
            self.doc_lengths = []
            self.avg_doc_length = 0
            self.doc_frequencies = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric characters
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _calculate_idf(self, word: str) -> float:
        """Calculate inverse document frequency"""
        doc_freq = self.doc_frequencies.get(word, 0)
        if doc_freq == 0:
            return 0
        
        total_docs = len(self.documents)
        # Standard IDF formula: log(total_docs / doc_freq)
        return math.log(total_docs / doc_freq)
    
    def _calculate_bm25_score(self, query_words: List[str], doc_index: int) -> float:
        """Calculate BM25 score for a document"""
        if doc_index >= len(self.documents):
            return 0
        
        doc_content = self.documents[doc_index].get('content', '')
        doc_words = self._tokenize(doc_content)
        doc_length = self.doc_lengths[doc_index]
        
        score = 0
        word_counts = Counter(doc_words)
        
        for word in query_words:
            if word in word_counts:
                tf = word_counts[word]
                idf = self._calculate_idf(word)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search using BM25 keyword matching"""
        if not self.documents:
            return []
        
        query_words = self._tokenize(query)
        if not query_words:
            return []
        
        # Calculate scores for all documents
        doc_scores = []
        for i in range(len(self.documents)):
            score = self._calculate_bm25_score(query_words, i)
            if score >= 0:  # Include zero scores for partial matches
                doc_scores.append((i, score))
        
        # Sort by score and return top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_index, score in doc_scores[:k]:
            doc = self.documents[doc_index]
            result = SearchResult(
                content=doc.get('content', ''),
                score=score,
                metadata=doc.get('metadata', {}),
                source="keyword",
                doc_id=doc.get('doc_id', f"doc_{doc_index}")
            )
            results.append(result)
        
        return results


class HybridRetriever:
    """Hybrid retrieval combining vector and keyword search"""
    
    def __init__(self, vector_store, embedding_provider, alpha: float = 0.7):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: Vector store for semantic search
            embedding_provider: Embedding provider for query encoding
            alpha: Weight for vector search (1-alpha for keyword search)
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.alpha = alpha
        self.keyword_searcher = KeywordSearcher()
        self.documents_indexed = False
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for both vector and keyword search"""
        # Prepare documents for keyword search
        keyword_docs = []
        for doc in documents:
            # Ensure metadata is a dictionary and create a copy to prevent modification issues
            metadata = doc.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}  # Convert non-dict metadata to empty dict
                
            keyword_docs.append({
                'content': doc.get('content', ''),
                'metadata': metadata,
                'doc_id': doc.get('doc_id', '')
            })
        
        self.keyword_searcher.add_documents(keyword_docs)
        self.documents_indexed = True
        logger.info("Documents indexed for hybrid retrieval")
    
    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores to 0-1 range"""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            for result in results:
                result.score = 1.0
        else:
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
    
    def _apply_document_type_boosts(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Apply boosts based on document classification and query intent"""
        query_lower = query.lower()
        
        # Detect query intent
        is_question_query = any(q in query_lower for q in ['what', 'how', 'when', 'where', 'why', 'who', 'explain', 'describe', '?'])
        is_technical_query = any(t in query_lower for t in ['algorithm', 'implement', 'code', 'programming', 'syntax', 'function', 'method'])
        is_curriculum_query = any(c in query_lower for c in ['syllabus', 'curriculum', 'course', 'lecture', 'semester', 'outline'])
        is_assignment_query = any(a in query_lower for a in ['assignment', 'homework', 'exercise', 'problem set', 'submission'])
        is_exam_query = any(e in query_lower for e in ['exam', 'test', 'quiz', 'midterm', 'final', 'question paper'])
        
        for result in results:
            boost_factor = 1.0
            metadata = result.metadata or {}
            
            # Boost documents that match the query intent
            if is_question_query and metadata.get('contains_questions', False):
                boost_factor *= 1.5
                
            if is_technical_query and metadata.get('content_type') == 'technical':
                boost_factor *= 1.4
                
            if is_curriculum_query and metadata.get('doc_category') == 'curriculum':
                boost_factor *= 1.6
                
            if is_assignment_query and metadata.get('doc_category') == 'assessment' and metadata.get('assessment_type') == 'assignment':
                boost_factor *= 1.7
                
            if is_exam_query and metadata.get('doc_category') == 'assessment' and metadata.get('assessment_type') == 'exam':
                boost_factor *= 1.8
            
            # Boost for question chunk types specifically
            if metadata.get('chunking_method') == 'question_based' and '?' in query_lower:
                boost_factor *= 1.5
            
            # Apply the boost
            result.score *= boost_factor
            
        return results
    
    def _combine_results(self, vector_results: List[SearchResult], 
                        keyword_results: List[SearchResult], 
                        query: str,
                        k: int) -> List[SearchResult]:
        """Combine and rank results from both search methods"""
        
        # Normalize scores
        vector_results = self._normalize_scores(vector_results)
        keyword_results = self._normalize_scores(keyword_results)
        
        # Apply document type boosts based on query intent
        vector_results = self._apply_document_type_boosts(vector_results, query)
        keyword_results = self._apply_document_type_boosts(keyword_results, query)
        
        # Create a map to combine results by document ID
        combined_scores = {}
        
        # Add vector search results
        for result in vector_results:
            doc_id = result.doc_id
            combined_scores[doc_id] = {
                'result': result,
                'vector_score': result.score,
                'keyword_score': 0.0
            }
        
        # Add keyword search results
        for result in keyword_results:
            doc_id = result.doc_id
            if doc_id in combined_scores:
                combined_scores[doc_id]['keyword_score'] = result.score
            else:
                combined_scores[doc_id] = {
                    'result': result,
                    'vector_score': 0.0,
                    'keyword_score': result.score
                }
        
        # Calculate hybrid scores
        final_results = []
        for doc_id, scores in combined_scores.items():
            result = scores['result']
            hybrid_score = (self.alpha * scores['vector_score'] + 
                          (1 - self.alpha) * scores['keyword_score'])
            
            # Apply decay factor to documents with very short content (likely incomplete chunks)
            if len(result.content) < 100:
                hybrid_score *= 0.7
            
            # Boost newer documents based on timestamp in metadata
            timestamp = None
            if result.metadata and 'timestamp' in result.metadata:
                try:
                    timestamp = float(result.metadata['timestamp'])
                    # Calculate recency boost (1.0 to 1.5 range)
                    import time
                    current_time = time.time()
                    # If document was added in the last hour, boost it more
                    time_diff_hours = (current_time - timestamp) / 3600
                    if time_diff_hours < 1:  # Less than an hour old
                        recency_boost = 1.5  # 50% boost
                    elif time_diff_hours < 24:  # Less than a day old
                        recency_boost = 1.3  # 30% boost
                    elif time_diff_hours < 72:  # Less than 3 days old
                        recency_boost = 1.2  # 20% boost
                    else:
                        recency_boost = 1.0  # No boost
                        
                    hybrid_score *= recency_boost
                    logger.debug(f"Applied recency boost of {recency_boost} to document {result.doc_id}")
                except (ValueError, TypeError):
                    pass
                    
            result.score = hybrid_score
            result.source = "hybrid"
            final_results.append(result)
        
        # Sort by hybrid score and return top k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:k]
    
    def search(self, query: str, k: int = 5, use_vector: bool = True, 
               use_keyword: bool = True, recency_boost: bool = False) -> List[SearchResult]:
        """
        Perform hybrid search
        
        Args:
            query: Search query
            k: Number of results to return
            use_vector: Whether to use vector search
            use_keyword: Whether to use keyword search
        """
        
        # Check if we have documents in the keyword searcher
        if not self.documents_indexed or not hasattr(self.keyword_searcher, 'documents') or not self.keyword_searcher.documents:
            # Try to initialize with documents from vector store
            try:
                all_docs = self.vector_store.get_all_documents()
                if all_docs:
                    logger.info(f"Auto-initializing hybrid retriever with {len(all_docs)} documents")
                    keyword_docs = []
                    for doc in all_docs:
                        keyword_docs.append({
                            'content': doc.content,
                            'metadata': doc.metadata or {},
                            'doc_id': doc.doc_id
                        })
                    if keyword_docs:
                        self.keyword_searcher.add_documents(keyword_docs)
                        self.documents_indexed = True
                    else:
                        # If we have documents but couldn't convert them properly, disable keyword search
                        use_keyword = False
                else:
                    # No documents in vector store, disable keyword search silently
                    use_keyword = False
            except Exception as e:
                logger.debug(f"Failed to auto-initialize hybrid search: {e}")
                use_keyword = False
            
        # If we still don't have documents indexed, fall back to vector search only
        if not self.documents_indexed or not hasattr(self.keyword_searcher, 'documents') or not self.keyword_searcher.documents:
            # Silently fallback to vector search without warning - it's expected behavior
            use_keyword = False
        
        vector_results = []
        keyword_results = []
        
        # Vector search
        if use_vector:
            try:
                query_embedding = self.embedding_provider.embed_text(query)
                vector_docs = self.vector_store.similarity_search(query_embedding, k=k*2)
                
                vector_results = [
                    SearchResult(
                        content=doc.content,
                        score=score,
                        metadata=doc.metadata or {},
                        source="vector",
                        doc_id=doc.doc_id
                    )
                    for doc, score in vector_docs
                ]
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
        
        # Keyword search
        if use_keyword:
            try:
                keyword_results = self.keyword_searcher.search(query, k=k*2)
            except Exception as e:
                logger.error(f"Keyword search failed: {e}")
        
        # Process results
        results = []
        
        if use_vector and use_keyword:
            results = self._combine_results(vector_results, keyword_results, query, k)
        elif use_vector:
            results = self._apply_document_type_boosts(vector_results[:k], query)
        elif use_keyword:
            results = self._apply_document_type_boosts(keyword_results[:k], query)
        
        # Apply recency boost if requested - this is additional to the boost already in _combine_results
        if recency_boost and results:
            logger.info("Applying additional recency boost to search results")
            for result in results:
                if result.metadata and 'timestamp' in result.metadata:
                    try:
                        timestamp = float(result.metadata['timestamp'])
                        import time
                        current_time = time.time()
                        # If document was added in the last day, boost it even more
                        time_diff_hours = (current_time - timestamp) / 3600
                        if time_diff_hours < 1:  # Less than an hour old
                            result.score *= 2.0  # Double the score for very recent docs
                        elif time_diff_hours < 24:  # Less than a day old
                            result.score *= 1.5  # 50% boost for recent docs
                    except (ValueError, TypeError):
                        pass
            
            # Re-sort after boosting
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:k]  # Limit to k results
        
        return results
    
    def explain_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Perform search and return detailed explanation
        """
        # Get individual results
        vector_results = []
        keyword_results = []
        
        if self.documents_indexed:
            try:
                query_embedding = self.embedding_provider.embed_text(query)
                vector_docs = self.vector_store.similarity_search(query_embedding, k=k*2)
                vector_results = [
                    SearchResult(
                        content=doc.content,
                        score=score,
                        metadata=doc.metadata or {},
                        source="vector",
                        doc_id=doc.doc_id
                    )
                    for doc, score in vector_docs
                ]
                # Apply document type boosts
                vector_results = self._apply_document_type_boosts(vector_results, query)
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
            
            try:
                keyword_results = self.keyword_searcher.search(query, k=k*2)
                # Apply document type boosts
                keyword_results = self._apply_document_type_boosts(keyword_results, query)
            except Exception as e:
                logger.error(f"Keyword search failed: {e}")
        
        # Get hybrid results
        hybrid_results = self._combine_results(vector_results, keyword_results, query, k)
        
        # Add more details to explanation
        query_analysis = {
            "is_question": "?" in query or any(q in query.lower() for q in ['what', 'how', 'when', 'where', 'why', 'who']),
            "likely_intent": self._analyze_query_intent(query),
            "key_terms": self._extract_key_terms(query)
        }
        
        return {
            "query": query,
            "query_analysis": query_analysis,
            "vector_results": vector_results[:k],
            "keyword_results": keyword_results[:k],
            "hybrid_results": hybrid_results,
            "alpha": self.alpha,
            "total_documents": len(self.keyword_searcher.documents)
        }
        
    def _analyze_query_intent(self, query: str) -> str:
        """Analyze the likely intent of the query"""
        query_lower = query.lower()
        
        if '?' in query or any(q in query_lower for q in ['what', 'how', 'when', 'where', 'why', 'who']):
            if any(t in query_lower for t in ['difference', 'compare', 'versus', 'vs']):
                return "comparison"
            elif any(t in query_lower for t in ['example', 'instance', 'illustration', 'show me']):
                return "example_seeking"
            elif any(t in query_lower for t in ['define', 'meaning', 'definition', 'what is']):
                return "definition_seeking"
            elif any(t in query_lower for t in ['step', 'how to', 'process', 'procedure']):
                return "procedural"
            else:
                return "factual_question"
        
        if any(c in query_lower for c in ['syllabus', 'curriculum', 'course']):
            return "curriculum_inquiry"
        
        if any(t in query_lower for t in ['assignment', 'homework', 'exercise', 'question']):
            return "assignment_related"
            
        return "general_information"
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from the query"""
        # Simple extraction of important words
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'by', 'with', 'about'}
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms[:5]  # Return top 5 key terms
