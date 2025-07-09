"""Adaptive chunking module with semantic splitting capabilities"""
import re
import spacy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import tiktoken
from loguru import logger

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    content: str
    start_index: int
    end_index: int
    chunk_id: str
    metadata: Dict[str, Any]
    semantic_score: Optional[float] = None
    
    def __post_init__(self):
        """Ensure metadata is always a dictionary with ChromaDB-compatible values"""
        if self.metadata is None:
            self.metadata = {}
        elif not isinstance(self.metadata, dict):
            # Handle non-dict metadata by storing it in a special key
            original = self.metadata
            self.metadata = {"_chunk_original_metadata": original}
            # Log this conversion for debugging
            logger.debug(f"Converted non-dict chunk metadata to dict: {type(original).__name__}")
        
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

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        pass

class FixedSizeChunker(ChunkingStrategy):
    """Traditional fixed-size chunking with overlap"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        if metadata is None:
            metadata = {}
        
        chunks = []
        tokens = self.encoding.encode(text)
        
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk = Chunk(
                content=chunk_text,
                start_index=i,
                end_index=i + len(chunk_tokens),
                chunk_id=f"chunk_{i}_{i + len(chunk_tokens)}",
                metadata={**metadata, "chunking_method": "fixed_size"}
            )
            chunks.append(chunk)
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """Semantic chunking based on sentence boundaries and coherence"""
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load spaCy model for sentence segmentation"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("spaCy model not found. Using simple sentence splitting (works well for most cases).")
            self.nlp = None
    
    def _get_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _detect_question_patterns(self, text: str) -> List[Dict]:
        """Detect question-answer patterns in the text"""
        questions = []
        
        # Regex patterns for common question formats
        patterns = [
            # Q1. What is...
            r'(?:^|\n)(?:Q|Question)\s*(\d+)[.:]?\s*([^\n]+)',
            # 1. What is...
            r'(?:^|\n)(\d+)[.)]?\s*([^\n]+\?)',
            # Simple question marks
            r'([^.!?\n]+\?)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) > 1:
                    # For numbered questions
                    question_num = match.group(1)
                    question_text = match.group(2)
                    questions.append({
                        'num': question_num,
                        'text': question_text,
                        'start': match.start(),
                        'end': match.end()
                    })
                else:
                    # For simple questions
                    questions.append({
                        'num': None,
                        'text': match.group(1),
                        'start': match.start(),
                        'end': match.end()
                    })
        
        return questions
    
    def _calculate_semantic_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate semantic similarity between sentences"""
        if self.nlp:
            doc1 = self.nlp(sent1)
            doc2 = self.nlp(sent2)
            return doc1.similarity(doc2)
        else:
            # Fallback to simple word overlap
            words1 = set(sent1.lower().split())
            words2 = set(sent2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
    
    def _convert_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata values to types compatible with ChromaDB"""
        safe_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, list):
                safe_metadata[k] = ",".join(map(str, v)) if v else ""
            elif isinstance(v, dict):
                # Convert nested dict to string
                safe_metadata[k] = str(v)
            else:
                # Simple types are fine
                safe_metadata[k] = v
        return safe_metadata
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        if metadata is None:
            metadata = {}
            
        # Convert metadata to ChromaDB-safe format
        safe_metadata = self._convert_metadata_for_chroma(metadata)
        
        # Special handling for documents with questions
        if safe_metadata.get('contains_questions') or 'question' in text.lower():
            return self._chunk_question_content(text, safe_metadata)
        
        sentences = self._get_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        current_topic = set()  # Track the current topic using important words - must be a set
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # Extract important words for topic continuity
            important_words = set()
            if self.nlp:
                doc = self.nlp(sentence)
                # Get nouns, verbs, and proper nouns as topic indicators
                important_words = {token.text.lower() for token in doc 
                                  if token.pos_ in ('NOUN', 'PROPN', 'VERB') and not token.is_stop}
            
            # Check topic continuity with current chunk
            topic_continuity = 0
            if current_topic and important_words:
                topic_overlap = current_topic.intersection(important_words)
                topic_continuity = len(topic_overlap) / max(len(current_topic), 1)
            
            # Start a new chunk if:
            # 1. Adding this sentence exceeds max size AND we have content already
            # 2. OR topic shift detected (low continuity) and minimum chunk size met
            new_chunk_needed = (
                (current_size + sentence_size > self.max_chunk_size and current_chunk) or
                (current_chunk and topic_continuity < 0.2 and current_size >= self.min_chunk_size)
            )
            
            if new_chunk_needed:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                # Create metadata with ChromaDB-compatible values
                chunk_metadata = {**safe_metadata, "chunking_method": "semantic"}
                # Convert topic_keywords from set to comma-separated string
                if current_topic:
                    chunk_metadata["topic_keywords"] = ",".join(list(current_topic)[:10])
                
                chunk = Chunk(
                    content=chunk_text,
                    start_index=len(chunks),
                    end_index=len(chunks) + 1,
                    chunk_id=f"semantic_chunk_{len(chunks)}",
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = [sentence]
                current_size = sentence_size
                current_topic = important_words
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                # Update topic with new important words
                if important_words:
                    current_topic.update(important_words)
                    # Keep topic focused on most recent content
                    if len(current_topic) > 30:
                        current_topic = set(list(current_topic)[-30:])
        
        # Add final chunk if it exists
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                # Create metadata with ChromaDB-compatible values
                chunk_metadata = {**safe_metadata, "chunking_method": "semantic"}
                # Convert topic_keywords from set to comma-separated string
                if current_topic:
                    chunk_metadata["topic_keywords"] = ",".join(list(current_topic)[:10])
                
                chunk = Chunk(
                    content=chunk_text,
                    start_index=len(chunks),
                    end_index=len(chunks) + 1,
                    chunk_id=f"semantic_chunk_{len(chunks)}",
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_question_content(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Special chunking method for question-answer content"""
        questions = self._detect_question_patterns(text)
        chunks = []
        
        # Ensure metadata is safe for ChromaDB
        safe_metadata = self._convert_metadata_for_chroma(metadata)
        
        if not questions:
            # Fall back to regular chunking if no questions detected
            return self.chunk(text, metadata)
        
        # Sort questions by position in text
        questions.sort(key=lambda q: q['start'])
        
        # Create chunks for each question and its context
        for i, question in enumerate(questions):
            # Determine chunk boundaries
            start_pos = question['start']
            end_pos = questions[i+1]['start'] if i < len(questions)-1 else len(text)
            
            # Get question context - include previous question if close enough
            context_start = questions[i-1]['end'] if i > 0 and question['start'] - questions[i-1]['end'] < 500 else start_pos
            
            # Extract chunk text
            chunk_text = text[context_start:end_pos].strip()
            
            # Ensure chunk isn't too large
            if len(chunk_text) > self.max_chunk_size * 1.5:  # Allow slightly larger chunks for questions
                # If too large, just include the current question and some context
                chunk_text = text[start_pos:start_pos + self.max_chunk_size].strip()
            
            # Create the chunk
            if chunk_text:
                # Create chunk metadata with ChromaDB-compatible values
                question_metadata = {
                    **safe_metadata, 
                    "chunking_method": "question_based"
                }
                
                # Format question number and text as strings
                if question.get('num'):
                    question_metadata["question_number"] = str(question.get('num'))
                    
                question_text = question.get('text', '')
                if question_text:
                    question_metadata["question_text"] = question_text[:100] + "..." if len(question_text) > 100 else question_text
                
                chunk = Chunk(
                    content=chunk_text,
                    start_index=start_pos,
                    end_index=start_pos + len(chunk_text),
                    chunk_id=f"question_chunk_{len(chunks)}",
                    metadata=question_metadata
                )
                chunks.append(chunk)
        
        return chunks


class HierarchicalChunker(ChunkingStrategy):
    """Hierarchical chunking that preserves document structure"""
    
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract sections based on headers and structure"""
        sections = []
        lines = text.split('\n')
        current_section = {"title": "", "content": [], "level": 0}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect headers (simple heuristic)
            if (line.isupper() or 
                re.match(r'^#+\s+', line) or  # Markdown headers
                re.match(r'^\d+\.\s+', line) or  # Numbered sections
                len(line) < 100 and line.endswith(':')):
                
                # Save previous section
                if current_section["content"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": line,
                    "content": [],
                    "level": self._detect_header_level(line)
                }
            else:
                current_section["content"].append(line)
        
        # Add final section
        if current_section["content"]:
            sections.append(current_section)
        
        return sections
    
    def _detect_header_level(self, line: str) -> int:
        """Detect header level"""
        if re.match(r'^#+\s+', line):
            return len(re.match(r'^#+', line).group())
        elif re.match(r'^\d+\.\s+', line):
            return 1
        elif line.isupper():
            return 1
        else:
            return 2
    
    def _convert_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata values to types compatible with ChromaDB"""
        safe_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, list):
                safe_metadata[k] = ",".join(map(str, v)) if v else ""
            elif isinstance(v, dict):
                # Convert nested dict to string
                safe_metadata[k] = str(v)
            else:
                # Simple types are fine
                safe_metadata[k] = v
        return safe_metadata
        
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        if metadata is None:
            metadata = {}
            
        # Convert metadata to ChromaDB-compatible format
        safe_metadata = self._convert_metadata_for_chroma(metadata)
        
        sections = self._extract_sections(text)
        chunks = []
        
        for section in sections:
            section_text = f"{section['title']}\n{' '.join(section['content'])}"
            
            if len(section_text) <= self.max_chunk_size:
                # Section fits in one chunk
                # Create metadata with ChromaDB-compatible values
                chunk_metadata = {
                    **safe_metadata,
                    "chunking_method": "hierarchical",
                    "section_title": section["title"],
                    "section_level": str(section["level"])  # Convert level to string for ChromaDB
                }
                
                chunk = Chunk(
                    content=section_text,
                    start_index=len(chunks),
                    end_index=len(chunks) + 1,
                    chunk_id=f"hierarchical_chunk_{len(chunks)}",
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
            else:
                # Split large section using semantic chunking
                semantic_chunker = SemanticChunker(self.max_chunk_size)
                
                # Create safe metadata for section
                section_metadata = {
                    **safe_metadata,
                    "section_title": section["title"],
                    "section_level": str(section["level"])  # Convert level to string
                }
                
                section_chunks = semantic_chunker.chunk(section_text, section_metadata)
                
                for chunk in section_chunks:
                    chunk.metadata["chunking_method"] = "hierarchical_semantic"
                    chunk.chunk_id = f"hierarchical_chunk_{len(chunks)}"
                    chunks.append(chunk)
        
        return chunks


class AdaptiveChunker:
    """Adaptive chunker that selects the best strategy based on content"""
    
    def __init__(self):
        self.strategies = {
            "fixed": FixedSizeChunker(),
            "semantic": SemanticChunker(),
            "hierarchical": HierarchicalChunker()
        }
    
    def _analyze_text_structure(self, text: str) -> str:
        """Analyze text to determine best chunking strategy"""
        lines = text.split('\n')
        
        # Count headers and structure elements
        header_count = 0
        list_count = 0
        paragraph_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if (re.match(r'^#+\s+', line) or  # Markdown headers
                re.match(r'^\d+\.\s+', line) or  # Numbered sections
                line.isupper() and len(line) < 100):
                header_count += 1
            elif re.match(r'^[-*•]\s+', line):  # Lists
                list_count += 1
            elif len(line) > 50:  # Likely paragraph
                paragraph_count += 1
        
        total_lines = len([l for l in lines if l.strip()])
        
        # Decision logic
        if header_count / total_lines > 0.1:  # More than 10% headers
            return "hierarchical"
        elif paragraph_count / total_lines > 0.5:  # Mostly paragraphs
            return "semantic"
        else:
            return "fixed"
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None, strategy: str = None) -> List[Chunk]:
        """Chunk text using adaptive strategy selection"""
        if strategy is None:
            strategy = self._analyze_text_structure(text)
        
        logger.info(f"Using {strategy} chunking strategy")
        
        if strategy in self.strategies:
            return self.strategies[strategy].chunk(text, metadata)
        else:
            logger.warning(f"Unknown strategy {strategy}, falling back to fixed")
            return self.strategies["fixed"].chunk(text, metadata)
