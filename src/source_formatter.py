"""Source formatting utilities to enhance source display"""

import os
from typing import Dict, Any, Optional, List
import re
from pathlib import Path
from urllib.parse import urlparse

class EnhancedSource:
    """Enhanced source information for better display in UI"""
    
    def __init__(self, source_info: Dict[str, Any]):
        self.source_info = source_info
        self.metadata = source_info.get("metadata", {})
        self.content = source_info.get("content", "")
        self.doc_id = source_info.get("doc_id", "")
        self.score = source_info.get("score", 0)
        self.source_path = self.metadata.get("source_file", "Unknown")
    
    @property
    def source_type(self) -> str:
        """Get the source type based on metadata or file extension"""
        # Check if doc_type is already in metadata
        doc_type = self.metadata.get("doc_type", "")
        
        if not doc_type:
            # Check for URL to identify web sources
            if self.metadata.get("url") or (self.source_path and (self.source_path.startswith("http://") or self.source_path.startswith("https://"))):
                return "web"
                
            # Try to determine from file extension
            if self.source_path and self.source_path != "Unknown":
                ext = Path(self.source_path).suffix.lower()
                if ext in ['.pdf']:
                    doc_type = "pdf"
                elif ext in ['.docx', '.doc']:
                    doc_type = "docx"
                elif ext in ['.txt', '.md', '.log']:
                    doc_type = "text"
                elif ext in ['.html', '.htm']:
                    doc_type = "html"
                elif ext in ['.pptx', '.ppt']:
                    doc_type = "presentation"
                elif ext in ['.xlsx', '.xls', '.csv']:
                    doc_type = "spreadsheet"
                elif ext in ['.json', '.xml', '.yaml', '.yml']:
                    doc_type = "data"
                elif ext in ['.py', '.js', '.java', '.cpp', '.cs']:
                    doc_type = "code"
                else:
                    doc_type = "document"
        
        return doc_type
    
    @property
    def source_icon(self) -> str:
        """Get an appropriate icon for the source type"""
        icon_map = {
            "pdf": "📄",
            "docx": "📝",
            "text": "📋",
            "html": "🌐",
            "presentation": "🖼️",
            "spreadsheet": "📊",
            "data": "📦",
            "code": "💻",
            "web": "🔗",
            "academic": "📚",
            "document": "📄",
            "curriculum": "📘",
            "assessment": "✅",
            "technical": "⚙️",
            "research": "🔬",
            "general": "📁"
        }
        return icon_map.get(self.source_type, "📄")
    
    @property
    def display_name(self) -> str:
        """Get a clean display name for the source"""
        if self.source_path and self.source_path != "Unknown":
            # For web URLs, use domain name instead of full URL
            if self.source_path.startswith("http://") or self.source_path.startswith("https://"):
                try:
                    parsed_url = urlparse(self.source_path)
                    return parsed_url.netloc
                except:
                    pass
            return os.path.basename(self.source_path)
        
        # Try to get title from metadata
        title = self.metadata.get("title", "")
        if title:
            return title
        
        return self.doc_id or "Unknown Source"
    
    @property
    def page_info(self) -> str:
        """Get page information if available"""
        page_info = ""
        
        # Check for PDF page marker in content
        page_match = re.search(r"--- Page (\d+) ---", self.content[:50])
        if page_match:
            page_num = page_match.group(1)
            page_info = f"Page {page_num}"
        
        # Or check if page number is in metadata
        elif "page" in self.metadata:
            page_info = f"Page {self.metadata['page']}"
        
        # For PDFs, check total pages
        total_pages = self.metadata.get("pages")
        if page_info and total_pages:
            page_info += f" of {total_pages}"
        
        return page_info
    
    @property
    def url(self) -> str:
        """Get URL if this is a web source"""
        url = self.metadata.get("url", "")
        
        # Check if source_path is a URL
        if not url and self.source_path and (self.source_path.startswith("http://") or self.source_path.startswith("https://")):
            url = self.source_path
        
        # Try to extract URL from content if it's HTML
        if not url and self.source_type == "html":
            url_match = re.search(r"(https?://[^\s]+)", self.content[:200])
            if url_match:
                url = url_match.group(1)
        
        return url
    
    @property
    def citation_info(self) -> str:
        """Get citation information for academic papers"""
        citation = ""
        
        # Check if academic metadata is available
        if self.metadata.get("author") and self.metadata.get("title"):
            author = self.metadata.get("author", "")
            title = self.metadata.get("title", "")
            year = self.metadata.get("year", "")
            journal = self.metadata.get("journal", "")
            
            # Format citation
            citation_parts = []
            if author:
                citation_parts.append(author)
            if year:
                citation_parts.append(f"({year})")
            if title:
                citation_parts.append(f'"{title}"')
            if journal:
                citation_parts.append(f"in {journal}")
                
            citation = ", ".join(citation_parts)
        
        return citation
    
    def format_for_display(self) -> Dict[str, Any]:
        """Format source information for display in UI"""
        display_info = {
            "source_id": self.doc_id,
            "name": self.display_name,
            "type": self.source_type,
            "icon": self.source_icon,
            "relevance_score": f"{self.score:.1%}" if isinstance(self.score, float) else self.score,
        }
        
        # Add page info if available
        page_info = self.page_info
        if page_info:
            display_info["page_info"] = page_info
        
        # Add URL if available
        url = self.url
        if url:
            display_info["url"] = url
        
        # Add citation if available
        citation = self.citation_info
        if citation:
            display_info["citation"] = citation
        
        # Add timestamp if available
        if "timestamp" in self.metadata:
            display_info["timestamp"] = self.metadata["timestamp"]
        
        # Add document classification if available
        if "doc_classification" in self.metadata:
            display_info["classification"] = self.metadata["doc_classification"]
        
        return display_info
    
    def __str__(self) -> str:
        """String representation of the source with enhanced information"""
        parts = [f"{self.source_icon} {self.display_name}"]
        
        if self.page_info:
            parts.append(f"({self.page_info})")
        
        if self.url:
            parts.append(f"- {self.url}")
        
        if self.citation_info:
            parts.append(f"- {self.citation_info}")
        
        return " ".join(parts)


def format_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format a list of sources with enhanced information"""
    enhanced_sources = []
    
    for source in sources:
        enhanced = EnhancedSource(source)
        enhanced_sources.append(enhanced.format_for_display())
    
    return enhanced_sources

def format_sources_as_markdown(sources: List[Dict[str, Any]]) -> List[str]:
    """Format a list of sources as markdown strings for display"""
    markdown_sources = []
    
    for source in sources:
        enhanced = EnhancedSource(source)
        markdown_sources.append(str(enhanced))
    
    return markdown_sources
