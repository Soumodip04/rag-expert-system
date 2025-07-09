"""Modern RAG Expert System UI"""
import streamlit as st
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
import json
import re
import os
import os
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

# Custom CSS for enhanced visual elements
def apply_custom_css():
    # Global CSS to ensure HTML is properly rendered
    st.write("""
    <style>
        /* Global fix for Streamlit HTML rendering issues */
        div.element-container div.stMarkdown div {
            overflow: visible !important;
        }
        
        /* Special styles for file pills and lists */
        .file-pill {
            background: linear-gradient(135deg, var(--color1, #4285F4), var(--color2, #34A853));
            color: white !important;
            padding: 0.4rem 0.7rem !important;
            border-radius: 12px !important;
            text-align: center !important;
            display: inline-flex !important;
            align-items: center !important;
            margin: 0.25rem !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        }
        
        /* File list styles */
        .file-list-item {
            padding: 8px 10px !important;
            display: flex !important;
            align-items: center !important;
            border-radius: 8px !important;
            margin-bottom: 8px !important;
        }
        
        /* General CSS Variables */
        :root {
            --transition-speed: 0.3s;
            --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
            --border-radius: 12px;
        }
        
        /* Enhanced File Type Pills */
        .file-pill {
            transition: all var(--transition-speed) ease;
            box-shadow: var(--box-shadow);
            transform: translateY(0);
        }
        .file-pill:hover {
            transform: translateY(-3px);
            box-shadow: 0 7px 14px rgba(0, 0, 0, 0.15), 0 3px 6px rgba(0, 0, 0, 0.1);
        }
        .file-count-badge {
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Enhanced File Type Group Boxes */
        .file-group-box {
            transition: all var(--transition-speed) ease;
            box-shadow: var(--box-shadow);
            transform: translateY(0);
            position: relative;
            overflow: hidden;
        }
        .file-group-box:hover {
            transform: translateY(-3px);
            box-shadow: 0 7px 14px rgba(0, 0, 0, 0.15), 0 3px 6px rgba(0, 0, 0, 0.1);
        }
        .file-group-box::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 80%);
            opacity: 0;
            transition: opacity var(--transition-speed) ease;
        }
        .file-group-box:hover::after {
            opacity: 1;
        }
        
        /* Enhanced File List */
        .file-list-item {
            transition: all var(--transition-speed) ease;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .file-list-item:hover {
            background-color: rgba(0, 0, 0, 0.03);
            transform: translateX(3px);
        }
        
        /* Status cards */
        .status-card {
            background: linear-gradient(135deg, #4285F4, #34A853);
            color: white;
            padding: 1rem;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            text-align: center;
            transition: all var(--transition-speed) ease;
        }
        .status-card:hover {
            box-shadow: 0 7px 14px rgba(0, 0, 0, 0.15), 0 3px 6px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        .status-number {
            font-size: 2rem;
            font-weight: 600;
        }
        .status-label {
            opacity: 0.9;
            font-size: 0.9rem;
        }
        
        /* Section headers with subtle decoration */
        .section-header {
            position: relative;
            padding-left: 1rem;
            font-weight: 600;
        }
        .section-header::before {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 4px;
            height: 20px;
            background: linear-gradient(to bottom, #4285F4, #34A853);
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

try:
    from src.rag_system import RAGExpertSystem
    from src.config import settings
    from src.document_processor import DocumentProcessingPipeline, ProcessedDocument
    from src.source_formatter import EnhancedSource, format_sources, format_sources_as_markdown
    from src.vector_stores import Document
    from src.chunking import Chunk
except ImportError:
    st.error("RAG system not found. Please ensure the src directory is available.")
    st.stop()

def initialize_session_state():
    """Initialize session state variables"""
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'current_domain' not in st.session_state:
        st.session_state.current_domain = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'document_count' not in st.session_state:
        st.session_state.document_count = 0
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

def create_header():
    """Create the main header"""
    st.markdown("""
    <div class="main-header">
        <div class="header-title">🧠 RAG Expert System</div>
        <div class="header-subtitle">Free • Private • Powerful AI Knowledge Assistant</div>
    </div>
    """, unsafe_allow_html=True)

def create_domain_selector():
    """Create domain selection interface"""
    st.markdown("### 🎯 Select Your Domain")
    
    domains = {
        "🏥 Healthcare": {
            "desc": "Medical documents, research papers, clinical guidelines",
            "icon": "🏥",
            "color": "linear-gradient(135deg, #20bf55 0%, #01baef 100%)"
        },
        "⚖️ Legal": {
            "desc": "Legal documents, contracts, regulations, case law",
            "icon": "⚖️",
            "color": "linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%)"
        },
        "🎓 Education": {
            "desc": "Academic papers, textbooks, course materials",
            "icon": "🎓",
            "color": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        },
        "💼 Business": {
            "desc": "Business documents, reports, policies, procedures",
            "icon": "💼",
            "color": "linear-gradient(135deg, #ff9966 0%, #ff5e62 100%)"
        },
        "🔬 Research": {
            "desc": "Scientific papers, research data, technical documentation",
            "icon": "🔬",
            "color": "linear-gradient(135deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%)"
        },
        "🌐 General": {
            "desc": "Any type of document or general knowledge",
            "icon": "🌐",
            "color": "linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%)"
        }
    }
    
    # Use CSS Grid for better responsive layout
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem;">
    """, unsafe_allow_html=True)
    
    for i, (domain, info) in enumerate(domains.items()):
        domain_name = domain.split(' ', 1)[1]
        is_selected = st.session_state.current_domain == domain
        border_style = "border: 2px solid #28a745;" if is_selected else "border: 1px solid rgba(49, 51, 63, 0.2);"
        
        # Create card with enhanced styling
        card_html = f"""
        <div 
            class="domain-button" 
            onclick="this.classList.add('clicked'); setTimeout(() => {{window.parent.postMessage({{type: 'streamlit:domainSelect', domain: '{domain}'}}, '*');}}, 150)"
            style="
                background: {info['color']};
                color: white;
                border-radius: 10px;
                padding: 1.2rem;
                margin-bottom: 1rem;
                cursor: pointer;
                text-align: center;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                {border_style}
                position: relative;
                overflow: hidden;
            "
            onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 10px 20px rgba(0,0,0,0.2)';"
            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';"
        >
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{info['icon']}</div>
            <div style="font-weight: bold; font-size: 1.2rem; margin-bottom: 0.5rem;">{domain_name}</div>
            <div style="font-size: 0.8rem; opacity: 0.9;">{info['desc']}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Hidden button for actual functionality
        if st.button(domain, key=f"domain_{i}"):
            st.session_state.current_domain = domain
            st.session_state.system_initialized = False
            st.rerun()
    
    # Close the grid container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show selected domain
    if st.session_state.current_domain:
        selected_domain = st.session_state.current_domain
        selected_info = domains.get(selected_domain, {"icon": "🎯", "color": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"})
        
        st.markdown(f"""
        <div style="
            background: {selected_info['color']};
            padding: 1rem;
            border-radius: 10px;
            color: white;
            display: flex;
            align-items: center;
            margin-top: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            animation: fadeIn 0.5s ease-out;
        ">
            <div style="font-size: 1.5rem; margin-right: 1rem;">{selected_info['icon']}</div>
            <div>
                <div style="font-weight: bold;">Active Domain</div>
                <div>{selected_domain}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def clean_domain_name(domain: str) -> str:
    """Clean domain name for use in vector store collection names"""
    import re
    # Remove emojis and special characters, keep only alphanumeric, dots, dashes, underscores
    cleaned = re.sub(r'[^\w\s-]', '', domain)
    # Replace spaces with underscores
    cleaned = cleaned.replace(' ', '_')
    # Remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    # Ensure it's not empty and has valid characters
    if not cleaned or not re.match(r'^[a-zA-Z0-9._-]+$', cleaned):
        cleaned = "general"
    return cleaned.lower()

def initialize_rag_system(domain: str):
    """Initialize RAG system for selected domain"""
    if not st.session_state.system_initialized:
        try:
            with st.spinner("🔄 Initializing RAG system..."):
                # Clean the domain name to remove emojis and spaces
                clean_domain = clean_domain_name(domain)
                
                # Create RAG system
                st.session_state.rag_system = RAGExpertSystem(domain=clean_domain)
                
                # Initialize the system
                initialization_result = st.session_state.rag_system.initialize()
                
                if initialization_result:
                    st.session_state.system_initialized = True
                    
                    # Get document count
                    update_document_count()
                    
                    st.success("✅ RAG system initialized successfully!")
                else:
                    st.error("❌ Failed to initialize RAG system - initialization returned False")
                    st.session_state.system_initialized = False
                    
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")
            st.session_state.system_initialized = False
            
            # Show more detailed error information
            with st.expander("🔍 Error Details", expanded=False):
                st.code(f"Error Type: {type(e).__name__}")
                st.code(f"Error Message: {str(e)}")
                
                # Check if it's a configuration issue
                if "api_key" in str(e).lower():
                    st.warning("💡 This looks like an API key configuration issue. Please check your .env file contains the correct API keys.")
                elif "model" in str(e).lower():
                    st.warning("💡 This looks like a model configuration issue. Please check your .env file has the correct model settings.")
                else:
                    st.info("💡 Try selecting a different domain or check that all required dependencies are installed.")

def create_chat_interface():
    """Create the main chat interface"""
    st.markdown("### 💬 Chat with Your Documents")
    
    # Show helpful message if no messages yet
    if not st.session_state.messages:
        st.info("👋 Welcome! Start by asking questions about your uploaded documents. You can also upload documents first using the file upload section on the right.")
    
    # Display chat messages using Streamlit's built-in chat components
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message.get("sources") and message["role"] == "assistant":
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        # Extract source information properly
                        if isinstance(source, dict):
                            # Get source file from metadata
                            source_path = source.get("metadata", {}).get("source_file", "Unknown")
                            confidence = source.get("score", 0)
                            doc_id = source.get("doc_id", "")
                            content_preview = source.get("content", "")
                            
                            # Clean up the source name (extract just the filename)
                            import os
                            source_name = os.path.basename(source_path) if source_path != "Unknown" else "Unknown"
                            
                            # Truncate content preview
                            if content_preview:
                                content_preview = content_preview[:200] + "..." if len(content_preview) > 200 else content_preview
                            
                            # Create a nice card-like display for each source
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                                border: 1px solid #dee2e6;
                                border-radius: 10px;
                                padding: 1rem;
                                margin: 0.5rem 0;
                                border-left: 4px solid #667eea;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                transition: all 0.3s ease;
                            ">
                                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                    <div style="font-size: 1.2rem; margin-right: 0.5rem;">{get_file_icon(source_name)}</div>
                                    <div style="font-weight: 600; color: #495057; margin: 0;">{source_name}</div>
                                </div>
                                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; font-size: 0.85rem;">
                                    <div style="background: #fff; padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid #dee2e6; color: #6c757d;">
                                        📋 Document ID: {doc_id}
                                    </div>
                                    <div style="background: #fff; padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid #dee2e6; color: #6c757d;">
                                        🎯 Relevance: {confidence:.1%}
                                    </div>
                                </div>
                                {"<div style='background: #fff; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem; font-size: 0.85em; color: #495057; border-left: 3px solid #28a745; max-height: 100px; overflow-y: auto;'><strong>📖 Content Preview:</strong><br>" + content_preview + "</div>" if content_preview else ""}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if confidence > 0:
                                    # Ensure confidence is within [0.0, 1.0] range for progress bar
                                    display_confidence = min(confidence, 1.0)
                                    
                                    # Color-coded relevance based on score
                                    if confidence >= 0.8:
                                        relevance_color = "🟢 Excellent"
                                    elif confidence >= 0.6:
                                        relevance_color = "🟡 Good"
                                    elif confidence >= 0.4:
                                        relevance_color = "🟠 Fair"
                                    else:
                                        relevance_color = "🔴 Low"
                                    
                                    st.progress(display_confidence, text=f"🎯 Relevance: {confidence:.1%} ({relevance_color})")
                            
                        else:
                            # Fallback for other source formats
                            source_name = str(source)
                            st.markdown(f"**{i}.** 📄 {source_name}")
            
            # Display timestamp
            if message.get("timestamp"):
                st.caption(f"⏰ {message['timestamp']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        handle_chat_input(prompt)

def handle_chat_input(prompt: str):
    """Handle chat input and generate response"""
    if not st.session_state.system_initialized:
        st.error("❌ Please select a domain first to initialize the system")
        return
    
    timestamp = datetime.now().strftime("%H:%M")
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Generate response using the actual RAG system
    try:
        with st.spinner("🤔 Searching documents and generating response..."):
            # Call query with just the prompt parameter to match the method signature
            # The second query method implementation takes only one argument
            response_data = st.session_state.rag_system.query(prompt)
            
            answer = response_data.answer if hasattr(response_data, 'answer') else response_data.get('answer', 'I could not find relevant information in the uploaded documents.')
            sources = response_data.sources if hasattr(response_data, 'sources') else response_data.get('sources', [])
            
            # Ensure sources are in the correct format for display
            formatted_sources = []
            
            # Format sources for display
            for source in sources:
                    if isinstance(source, dict):
                        # Source is already in the correct format
                        formatted_sources.append(source)
                    else:
                        # Convert other formats to dict
                        formatted_sources.append({
                            "metadata": {"source_file": str(source)},
                            "score": 0,
                            "doc_id": ""
                        })
            
            # Try to enhance answer with source formatting
            try:
                # Format markdown for any citations in the answer
                markdown_sources = format_sources_as_markdown(formatted_sources)
                
                # Check if answer has citations like [1], [2], etc.
                enhanced_answer = answer
                citation_pattern = r'\[(\d+)\]'
                citations = re.findall(citation_pattern, answer)
                
                if citations and markdown_sources:
                    # Add formatted citations at the end of the answer
                    enhanced_answer += "\n\n**Sources:**\n"
                    for i, citation_num in enumerate(citations):
                        if i < len(markdown_sources):
                            enhanced_answer += f"{citation_num}. {markdown_sources[i]}\n"
                
                answer = enhanced_answer
            except Exception as format_error:
                # If formatting fails, just use the original answer
                pass
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": formatted_sources,
                "timestamp": timestamp
            })
    
    except Exception as e:
        error_msg = f"❌ I encountered an error: {str(e)}"
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "timestamp": timestamp
        })
    
    # Rerun to update the interface
    st.rerun()

def create_system_status():
    """Create system status panel"""
    st.markdown("### 📊 System Status")
    
    # Status metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="status-card">
            <div class="status-number">{st.session_state.document_count}</div>
            <div class="status-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        chat_count = len(st.session_state.messages) // 2
        st.markdown(f"""
        <div class="status-card">
            <div class="status-number">{chat_count}</div>
            <div class="status-label">Conversations</div>
        </div>
        """, unsafe_allow_html=True)
    
    # System info
    if st.session_state.system_initialized:
        st.success("🟢 System Online")
        st.info(f"🎯 Active Domain: {st.session_state.current_domain}")
    else:
        st.warning("🟡 System Offline")

def create_file_upload():
    """Create file upload interface"""
    st.markdown("### 📁 Upload Documents")
    
    # Define file type groups with icons
    file_type_groups = {
        "Documents": {
            "icon": "📝",
            "types": ['txt', 'rtf', 'odt'],
            "color": "#4285F4"
        },
        "PDFs": {
            "icon": "📕",
            "types": ['pdf'],
            "color": "#FF0000"
        },
        "Word Documents": {
            "icon": "📄",
            "types": ['docx', 'doc'],
            "color": "#2B579A"
        },
        "Spreadsheets": {
            "icon": "📊",
            "types": ['xlsx', 'xls', 'csv', 'ods'],
            "color": "#34A853"
        },
        "Presentations": {
            "icon": "🖼️",
            "types": ['pptx', 'ppt', 'odp'],
            "color": "#FBBC05"
        },
        "Web & Markup": {
            "icon": "🌐",
            "types": ['html', 'htm', 'xml', 'md', 'rst'],
            "color": "#EA4335"
        },
        "Data formats": {
            "icon": "📦",
            "types": ['json', 'yaml', 'yml', 'toml', 'ini'],
            "color": "#4285F4"
        },
        "Programming": {
            "icon": "💻",
            "types": ['py', 'js', 'ts', 'java', 'cpp', 'c', 'h', 'css', 'sql'],
            "color": "#7B68EE"
        },
        "E-books": {
            "icon": "📚",
            "types": ['epub', 'tex', 'bib'],
            "color": "#FF5722"
        },
        "Archives": {
            "icon": "🗃️",
            "types": ['zip', 'tar', 'gz'],
            "color": "#795548"
        }
    }
    
    # Flatten the types list for the file_uploader
    all_types = []
    for group in file_type_groups.values():
        all_types.extend(group["types"])
    
    # Create file type category boxes displayed side by side
    cols = st.columns(5)  # Display 5 boxes per row
    
    for i, (group_name, group_info) in enumerate(file_type_groups.items()):
        # Calculate number of files of this type
        file_count = 0
        if 'uploaded_files' in locals() or 'uploaded_files' in globals():
            if uploaded_files:
                for file in uploaded_files:
                    ext = file.name.split('.')[-1].lower()
                    if ext in group_info["types"]:
                        file_count += 1
        
        # Calculate animation delay for staggered appearance
        delay = i * 0.08
        
        # Calculate darker color for gradient
        try:
            darker_color = adjust_color_brightness(group_info["color"], -20)
        except:
            darker_color = group_info["color"]
        
        # Display in columns for side-by-side layout
        with cols[i % 5]:  # Wrap to next row after every 5 items
            st.markdown(
                f"""
                <div class="file-group-box" style="
                    background: linear-gradient(135deg, {group_info["color"]}, {darker_color});
                    color: white;
                    padding: 0.8rem 0.5rem;
                    border-radius: 15px;
                    text-align: center;
                    margin-bottom: 0.8rem;
                    height: 90px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    animation: fadeIn 0.6s ease forwards {delay}s;
                    opacity: 0;
                ">
                    <div style="font-size: 1.7rem; margin-bottom: 0.3rem; filter: drop-shadow(0 2px 3px rgba(0,0,0,0.2));">{group_info["icon"]}</div>
                    <div style="font-size: 0.9rem; font-weight: 500;">{group_name}</div>
                    <div style="
                        background: rgba(255,255,255,0.25); 
                        padding: 0.1rem 0.6rem; 
                        border-radius: 12px; 
                        font-size: 0.85rem; 
                        font-weight: 600;
                        margin-top: 0.3rem;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        backdrop-filter: blur(5px);
                    ">{file_count}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("""
    <style>
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .file-pill {
        display: inline-flex !important;
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
    
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .stMarkdown div {
        overflow: visible !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add button to load default knowledge base
    if st.session_state.system_initialized:
        col1, col2 = st.columns([3, 2])
        with col1:
            if st.button("🎯 Load Default Knowledge Base", type="secondary", use_container_width=True):
                load_default_knowledge_base()
        with col2:
            if st.button("🧹 Clear All Documents", type="primary", use_container_width=True):
                clear_documents()
        st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        accept_multiple_files=True,
        type=all_types,
        help="📄 Supported formats: Office docs (PDF, DOCX, XLSX, PPTX), Web (HTML, XML), Data (JSON, CSV, YAML), Code files (PY, JS), E-books (EPUB), and more!"
    )
    
    if uploaded_files:
        # Count files by type
        file_counts = {}
        total_size = 0
        for file in uploaded_files:
            ext = file.name.split('.')[-1].lower()
            if ext in file_counts:
                file_counts[ext] += 1
            else:
                file_counts[ext] = 1
            total_size += file.size
            
        # Show file summary with enhanced visual indicators
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"""
            <div class="status-card" style="
                background: linear-gradient(135deg, #4285F4, #34A853);
                border-radius: 15px;
                padding: 1.2rem;
                position: relative;
                overflow: hidden;
            ">
                <div style="position: absolute; top: -30px; right: -30px; width: 100px; height: 100px; background: rgba(255,255,255,0.1); border-radius: 50%;"></div>
                <div style="position: absolute; bottom: -20px; left: -20px; width: 70px; height: 70px; background: rgba(255,255,255,0.08); border-radius: 50%;"></div>
                <div class="status-number" style="position: relative; font-size: 2.2rem; font-weight: 700;">{len(uploaded_files)}</div>
                <div class="status-label" style="position: relative; opacity: 0.9; font-size: 0.95rem; margin-top: 5px;">Files Selected</div>
            </div>
            """, unsafe_allow_html=True)
            
        with cols[1]:
            if total_size > 1024 * 1024:
                size_str = f"{total_size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{total_size / 1024:.1f} KB"
                
            st.markdown(f"""
            <div class="status-card" style="
                background: linear-gradient(135deg, #FF5722, #FF9800);
                border-radius: 15px;
                padding: 1.2rem;
                position: relative;
                overflow: hidden;
            ">
                <div style="position: absolute; top: -30px; right: -30px; width: 100px; height: 100px; background: rgba(255,255,255,0.1); border-radius: 50%;"></div>
                <div style="position: absolute; bottom: -20px; left: -20px; width: 70px; height: 70px; background: rgba(255,255,255,0.08); border-radius: 50%;"></div>
                <div class="status-number" style="position: relative; font-size: 2.2rem; font-weight: 700;">{size_str}</div>
                <div class="status-label" style="position: relative; opacity: 0.9; font-size: 0.95rem; margin-top: 5px;">Total Size</div>
            </div>
            """, unsafe_allow_html=True)
        
        # File type breakdown header
        st.markdown("""
        <div style="margin-bottom: 8px;">
            <div style="
                display: flex;
                align-items: center;
                gap: 8px;
            ">
                <div style="
                    width: 4px; 
                    height: 20px; 
                    background: linear-gradient(to bottom, #4285F4, #0F9D58);
                    border-radius: 2px;
                    margin-right: 8px;
                "></div>
                <h3 style="margin: 0; color: white; font-weight: 500; font-size: 20px;">File Type Breakdown</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        sorted_exts = sorted(file_counts.items(), key=lambda x: x[0])
        
        st.write('''
        <style>
        .side-by-side-container {
            display: flex !important;
            flex-direction: column !important;
            gap: 0px !important;
            margin: 5px 0 !important;
            width: 100% !important;
        }
        
        .file-types-column {
            width: auto !important;
            background: #1a1a27 !important; 
            padding: 0 !important;
            box-shadow: none !important;
        }
        
        .file-list-column {
            width: 100% !important;
            background: #1d1d2b !important;
            border-radius: 10px !important;
            padding: 15px 20px !important;
            box-shadow: 0 3px 6px rgba(0,0,0,0.2) !important;
            max-height: 500px !important;
            overflow-y: auto !important;
            color: white !important;
            margin-top: 10px !important;
        }
        
        .file-type-pills-container {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: wrap !important;
            gap: 10px !important;
            width: 100% !important;
            margin-top: 10px !important;
            margin-bottom: 10px !important;
        }
        </style>
        <div class="side-by-side-container">
        <div class="file-types-column">
            <div class="file-type-pills-container">
        ''', unsafe_allow_html=True)
        
        for ext, count in sorted_exts:
            if ext == 'pdf':
                bg_color = "#FF0000"
                icon = "📕"
            elif ext == 'py':
                bg_color = "#7B68EE"
                icon = "💻"
            elif ext == 'xlsx':
                bg_color = "#34A853"
                icon = "📊"
            else:
                bg_color = "#6c757d"
                icon = get_file_icon(f"file.{ext}")
                
                for group_info in file_type_groups.values():
                    if ext in group_info["types"]:
                        bg_color = group_info["color"]
                        break
            st.write(f'''
                <div style="
                    background: {bg_color}; 
                    color: white; 
                    padding: 10px 15px;
                    border-radius: 50px; 
                    display: flex; 
                    align-items: center;
                    justify-content: space-between;
                    width: 120px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    margin: 0 0 8px 0;">
                    <div style="
                        display: flex;
                        align-items: center;
                    ">
                        <span style="margin-right: 6px; font-size: 16px;">{icon}</span>
                        <span style="font-weight: 600; font-size: 15px;">.{ext}</span>
                    </div>
                    <div style="
                        background: rgba(255,255,255,0.25);
                        color: white;
                        height: 24px;
                        width: 24px;
                        font-size: 13px;
                        font-weight: 600;
                        border-radius: 50%;
                        margin-left: 8px;
                        display: flex;
                        align-items: center;
                        justify-content: center;">{count}</div>
                </div>
            ''', unsafe_allow_html=True)
        
        st.write('</div></div></div>', unsafe_allow_html=True)
        
        st.write('''
        <div class="file-list-column">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding: 8px 0;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 16px; margin-right: 8px;">📋</span>
                    <h4 style="margin: 0; font-weight: 500; color: white;">Detailed File List</h4>
                </div>
                <div style="width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 16px;">▲</span>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Render each file individually for maximum compatibility
        for i, file in enumerate(uploaded_files):
                file_size = file.size
                if file_size > 1024 * 1024:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{file_size / 1024:.1f} KB"
                
                ext = file.name.split('.')[-1].lower() if '.' in file.name else ""
                file_color = "#6c757d"
                
                if ext == "pdf":
                    file_color = "#FF0000"
                    file_icon = "📕"
                elif ext == "py":
                    file_color = "#7B68EE"
                    file_icon = "💻"
                elif ext == "xlsx":
                    file_color = "#34A853"
                    file_icon = "📊"
                else:
                    for group_info in file_type_groups.values():
                        if ext in group_info["types"]:
                            file_color = group_info["color"]
                            break
                    file_icon = get_file_icon(file.name)
                st.write(f'''
                <div style="
                    padding: 20px;
                    display: flex;
                    align-items: center;
                    border-radius: 5px;
                    margin-bottom: 8px;
                    background: white;
                    box-shadow: none;
                    border-bottom: 1px solid #eee;">
                    <div style="
                        background: {file_color};
                        color: white;
                        width: 36px;
                        height: 36px;
                        border-radius: 4px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-right: 15px;
                        font-size: 18px;
                    ">
                        {file_icon}
                    </div>
                    <div style="flex-grow: 1; display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-weight: normal; color: #333; font-size: 0.95rem; margin-right: 10px; max-width: 70%; overflow: hidden; text-overflow: ellipsis;">{file.name}</div>
                        <div style="
                            font-size: 0.85rem;
                            color: #888;
                        ">{size_str}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        # Close the file list column and container
        st.write('''
            </div>
        </div>
        <div style="margin-top: 20px;"></div>
        ''', unsafe_allow_html=True)
        
        if st.session_state.system_initialized:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Process Documents", type="primary"):
                    process_uploaded_files(uploaded_files)
            
            with col2:
                if st.button("📚 Load Default Knowledge Base", type="secondary"):
                    load_default_knowledge_base()
        else:
            st.warning("⚠️ Please initialize a domain first.")
    else:
        st.info("📤 Please select documents to upload to your knowledge base.")

def process_uploaded_files(uploaded_files):
    """Process uploaded files with enhanced progress tracking"""
    if not st.session_state.system_initialized:
        st.error("❌ Please initialize a domain first")
        return
    
    # Create container for processing UI
    processing_container = st.container()
    
    with processing_container:
        st.markdown("### 🔄 Processing Documents")
        
        total_files = len(uploaded_files)
        success_count = 0
        failed_files = []
        file_results = {}
        start_time = time.time()
        
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_cols = st.columns(3)
            file_status = st.empty()
            
            with metrics_cols[0]:
                processed_metric = st.empty()
            with metrics_cols[1]:
                success_metric = st.empty()
            with metrics_cols[2]:
                time_metric = st.empty()
            
            # Initialize pipeline for document processing if not using RAG system's pipeline
            pipeline = DocumentProcessingPipeline()
            
            # Update metrics function
            def update_metrics(idx, success, failed):
                progress = (idx + 1) / total_files
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1) if idx > 0 else 0
                remaining = avg_time * (total_files - idx - 1)
                
                progress_bar.progress(progress)
                processed_metric.markdown(f"""
                <div class="status-card">
                    <div class="status-number">{idx + 1}/{total_files}</div>
                    <div class="status-label">Files Processed</div>
                </div>
                """, unsafe_allow_html=True)
                
                success_metric.markdown(f"""
                <div class="status-card">
                    <div class="status-number">{success}</div>
                    <div class="status-label">Successfully Processed</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Format time remaining
                if remaining < 60:
                    time_text = f"{remaining:.1f}s"
                else:
                    time_text = f"{remaining/60:.1f}m"
                    
                time_metric.markdown(f"""
                <div class="status-card">
                    <div class="status-number">{time_text}</div>
                    <div class="status-label">Time Remaining</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Process each file
            for idx, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                status_text.markdown(f"<h4>🔄 Processing {file_name}...</h4>", unsafe_allow_html=True)
                
                # Create temp directory
                os.makedirs("data/temp", exist_ok=True)
                temp_path = f"data/temp/{file_name}"
                
                try:
                    # Save file temporarily
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process with RAG system
                    file_start_time = time.time()
                    
                    # Try to get file type for better handling
                    file_ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
                    
                    # Add more detailed error handling for PDFs
                    if file_ext == 'pdf':
                        try:
                            # Ensure PyPDF2 is available
                            import PyPDF2
                            # Verify the PDF can be opened
                            with open(temp_path, 'rb') as test_file:
                                pdf_reader = PyPDF2.PdfReader(test_file)
                                page_count = len(pdf_reader.pages)
                                
                                # Try to extract text to verify it's not empty or corrupt
                                sample_text = ""
                                try:
                                    if pdf_reader.pages and len(pdf_reader.pages) > 0:
                                        # Try first page
                                        first_page = pdf_reader.pages[0]
                                        sample_text = first_page.extract_text() or ""
                                        if not sample_text.strip():
                                            # Try extracting a few more pages
                                            for i in range(1, min(3, page_count)):
                                                page_text = pdf_reader.pages[i].extract_text() or ""
                                                if page_text.strip():
                                                    sample_text = page_text
                                                    break
                                except Exception as ext_err:
                                    # Log extraction error but continue - we'll handle with fallback
                                    st.warning(f"PDF text extraction warning: {str(ext_err)}")
                                
                                # Add PDF metadata to status
                                file_status.markdown(f"<small>📄 PDF with {page_count} pages</small>", unsafe_allow_html=True)
                                
                                # If we couldn't extract any text, issue a warning
                                if not sample_text.strip():
                                    file_status.markdown("<small>⚠️ No text detected in PDF. May be a scanned document.</small>", unsafe_allow_html=True)
                                    
                        except Exception as pdf_err:
                            error_msg = f"PDF pre-validation error: {str(pdf_err)}"
                            status_text.markdown(f"<h4>❌ Error with PDF file: {file_name}</h4>", unsafe_allow_html=True)
                            file_status.markdown(f"<small>Error: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}</small>", 
                                                unsafe_allow_html=True)
                            raise Exception(error_msg)
                    
                    # Use RAG system to add document with diagnostics
                    try:
                        # For PDFs, use a more robust approach with direct pipeline processing
                        if file_ext == 'pdf':
                            # First try processing with our custom pipeline to better diagnose issues
                            with st.spinner("Processing PDF content..."):
                                processed_doc = pipeline.process_file(temp_path, {"source": "user_upload"})
                                
                                if processed_doc and processed_doc.content and len(processed_doc.content.strip()) > 50:
                                    # If we got content from the PDF, manually add it to the RAG system
                                    status_text.markdown("<h4>✅ PDF content extracted successfully</h4>", unsafe_allow_html=True)
                                    file_status.markdown(f"<small>Content length: {len(processed_doc.content)} characters</small>", unsafe_allow_html=True)
                                    
                                    # Try to use our direct add method for better control
                                    try:
                                        # First try our enhanced direct addition
                                        direct_result = direct_add_processed_document(processed_doc, temp_path)
                                        if not direct_result:
                                            # If direct addition fails, try using the standard method
                                            status_text.markdown("<h4>⚠️ Using fallback document processing method</h4>", unsafe_allow_html=True)
                                            processing_result = st.session_state.rag_system.add_documents([temp_path])
                                        if direct_result:
                                            processing_result = True
                                        else:
                                            # Fallback to standard method
                                            processing_result = st.session_state.rag_system.add_documents([temp_path])
                                    except:
                                        # If direct add fails, use the standard method
                                        processing_result = st.session_state.rag_system.add_documents([temp_path])
                                else:
                                    # Failed to get meaningful content
                                    error_msg = "Could not extract meaningful text content from PDF"
                                    status_text.markdown(f"<h4>❌ PDF content extraction failed: {file_name}</h4>", unsafe_allow_html=True)
                                    file_status.markdown(f"<small>Error: PDF may be scanned/image-based or encrypted</small>", 
                                                      unsafe_allow_html=True)
                                    # Try one more time with standard method before giving up
                                    processing_result = st.session_state.rag_system.add_documents([temp_path])
                        else:
                            # For non-PDF files, process normally
                            processing_result = st.session_state.rag_system.add_documents([temp_path])
                    except Exception as proc_err:
                        status_text.markdown(f"<h4>❌ Processing error: {file_name}</h4>", unsafe_allow_html=True)
                        error_msg = f"Processing error: {str(proc_err)}"
                        file_status.markdown(f"<small>{error_msg[:100]}{'...' if len(error_msg) > 100 else ''}</small>", 
                                            unsafe_allow_html=True)
                        raise Exception(error_msg)
                    
                    # Calculate processing time
                    processing_time = time.time() - file_start_time
                    
                    if processing_result:
                        success_count += 1
                        file_results[file_name] = {
                            "status": "success",
                            "time": processing_time,
                            "type": file_ext
                        }
                        status_text.markdown(f"<h4>✅ Processed {file_name}</h4>", unsafe_allow_html=True)
                    else:
                        failed_files.append(file_name)
                        file_results[file_name] = {
                            "status": "failed",
                            "time": processing_time,
                            "type": file_ext,
                            "error": "Processing returned False"
                        }
                        status_text.markdown(f"<h4>❌ Failed to process {file_name}</h4>", unsafe_allow_html=True)
                    
                except Exception as e:
                    error_msg = str(e)
                    failed_files.append(file_name)
                    file_results[file_name] = {
                        "status": "error",
                        "time": time.time() - file_start_time if 'file_start_time' in locals() else 0,
                        "type": file_name.split('.')[-1].lower() if '.' in file_name else '',
                        "error": error_msg
                    }
                    status_text.markdown(f"<h4>❌ Error processing {file_name}</h4>", unsafe_allow_html=True)
                    
                    # Show error details in small text
                    file_status.markdown(f"<small>Error: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}</small>", 
                                        unsafe_allow_html=True)
                
                finally:
                    # Clean up
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                
                # Update metrics after each file
                update_metrics(idx, success_count, len(failed_files))
                time.sleep(0.1)  # Brief pause for better UX
            
            # Update document count using the dedicated function
            # This ensures accurate count by fetching directly from vector store
            update_document_count()
            
            # Show status message based on success
            if success_count == total_files:
                status_text.markdown("""
                <div style="background: linear-gradient(135deg, #20bf55 0%, #01baef 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center; color: white;
                            margin: 1rem 0; animation: fadeIn 0.5s ease-out;">
                    <h3>🎉 All files processed successfully!</h3>
                    <p>Your documents are now ready to be queried.</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            elif success_count > 0:
                status_text.markdown(f"""
                <div style="background: linear-gradient(135deg, #f9d423 0%, #ff4e50 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center; color: white;
                            margin: 1rem 0; animation: fadeIn 0.5s ease-out;">
                    <h3>⚠️ Processed {success_count} out of {total_files} files</h3>
                    <p>Some files could not be processed. Check the details below.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                status_text.markdown("""
                <div style="background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center; color: white;
                            margin: 1rem 0; animation: fadeIn 0.5s ease-out;">
                    <h3>❌ No files were processed successfully</h3>
                    <p>Please check the file formats and try again.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show processing summary
            st.markdown("### 📊 Processing Summary")
            summary_cols = st.columns(3)
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Format time nicely
            if total_time < 60:
                time_format = f"{total_time:.1f} seconds"
            elif total_time < 3600:
                time_format = f"{total_time/60:.1f} minutes"
            else:
                time_format = f"{total_time/3600:.1f} hours"
            
            with summary_cols[0]:
                st.metric("Files Processed", f"{total_files}")
            with summary_cols[1]:
                st.metric("Success Rate", f"{(success_count/total_files)*100:.1f}%")
            with summary_cols[2]:
                st.metric("Total Time", time_format)
            
            # Show detailed results in expander
            if failed_files:
                with st.expander("❌ Failed Files", expanded=True):
                    for file_name in failed_files:
                        result = file_results.get(file_name, {})
                        error_msg = result.get("error", "Unknown error")
                        st.markdown(f"• **{file_name}**: {error_msg}")
                        
            # Show successful files
            if success_count > 0:
                with st.expander("✅ Successfully Processed Files", expanded=False):
                    for file_name, result in file_results.items():
                        if result["status"] == "success":
                            proc_time = result["time"]
                            if proc_time < 1:
                                time_str = f"{proc_time*1000:.0f} ms"
                            else:
                                time_str = f"{proc_time:.1f} s"
                            st.markdown(f"• **{file_name}** (processed in {time_str})")
            
            # Show user what to do next
            if success_count > 0:
                st.markdown("""
                ### 🚀 What's Next?
                
                You can now ask questions about your documents in the chat interface. 
                Try asking specific questions or requesting summaries of the content.
                """)
                
                # Sample questions based on document types
                if any(result.get("type") == "pdf" for result in file_results.values()):
                    st.info("📄 For PDF documents, try asking: 'What are the main topics covered in the PDFs?'")
                if any(result.get("type") in ["docx", "doc"] for result in file_results.values()):
                    st.info("📝 For Word documents, try asking: 'Can you summarize the key points in the Word documents?'")
                if any(result.get("type") in ["csv", "xlsx", "xls"] for result in file_results.values()):
                    st.info("📊 For spreadsheets, try asking: 'What trends or patterns can you identify in the data?'")
                if any(result.get("type") in ["py", "js", "java", "cpp"] for result in file_results.values()):
                    st.info("💻 For code files, try asking: 'Explain the functionality of this code' or 'How can I improve this code?'")
            
            # Reset file_status container
            file_status.empty()
            
            # If no files were processed successfully, show error message
            if success_count == 0:
                st.error("❌ Failed to process any files. Please check file formats and try again.")
            
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()
            
            if success_count > 0:
                st.rerun()
            
        except Exception as e:
            st.error(f"❌ Unexpected error during processing: {str(e)}")
            progress_bar.empty()
            status_text.empty()

def load_default_knowledge_base():
    """Load predefined knowledge base for the current domain"""
    if not st.session_state.system_initialized:
        st.error("❌ Please initialize a domain first")
        return
    
    # Map domains to their default knowledge files
    domain_files = {
        "healthcare": ["data/healthcare_diabetes_guidelines.txt"],
        "legal": ["data/legal_employment_contracts.txt"],
        "financial": ["data/financial_risk_assessment.txt"],
        "business": ["data/financial_risk_assessment.txt"],
        "general": ["data/healthcare_diabetes_guidelines.txt", "data/legal_employment_contracts.txt", "data/financial_risk_assessment.txt"]
    }
    
    clean_domain = clean_domain_name(st.session_state.current_domain)
    files_to_load = domain_files.get(clean_domain, [])
    
    if not files_to_load:
        st.warning(f"No default knowledge base available for {st.session_state.current_domain}")
        return
    
    # Filter files that actually exist
    existing_files = [f for f in files_to_load if os.path.exists(f)]
    
    if not existing_files:
        st.error("❌ Default knowledge base files not found")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for idx, file_path in enumerate(existing_files):
            status_text.text(f"🔄 Loading {os.path.basename(file_path)}...")
            
            if st.session_state.rag_system.add_documents([file_path]):
                status_text.text(f"✅ Loaded {os.path.basename(file_path)}")
            else:
                status_text.text(f"❌ Failed to load {os.path.basename(file_path)}")
            
            progress = (idx + 1) / len(existing_files)
            progress_bar.progress(progress)
            time.sleep(0.5)
        
        progress_bar.progress(1.0)
        status_text.text("🎉 Default knowledge base loaded!")
        st.success(f"✅ Loaded {len(existing_files)} default knowledge files!")
        st.balloons()
        
        # Update document count using the dedicated function
        # This ensures accurate count by fetching directly from vector store
        update_document_count()
    except Exception as e:
        st.error(f"❌ Error loading default knowledge base: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def create_analytics():
    """Create analytics dashboard"""
    st.markdown("### 📈 Analytics Dashboard")
    
    # Create sample analytics (you can replace with real data)
    col1, col2 = st.columns(2)
    
    with col1:
        # Display a note indicating this is demo data
        st.info("📊 The charts below show sample demonstration data from January 2024 and do not reflect current system usage.")
        
        # Query frequency chart using graph_objects instead of px
        # Generate date strings manually
        start_date = datetime(2024, 1, 1)
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
        queries = [5, 8, 12, 15, 10, 20, 18]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, 
            y=queries,
            mode='lines+markers',
            name='Queries'
        ))
        fig.update_layout(
            title="Daily Query Volume",
            xaxis_title="Date",
            yaxis_title="Queries",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Document types pie chart using graph_objects instead of px
        doc_types = ['PDF', 'TXT', 'DOCX', 'JSON', 'HTML']
        doc_counts = [15, 8, 12, 5, 3]
        
        fig = go.Figure(data=[go.Pie(
            labels=doc_types,
            values=doc_counts,
        )])
        fig.update_layout(
            title="Document Types",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

def clear_documents():
    """Clear all uploaded documents"""
    if not st.session_state.system_initialized:
        st.error("❌ System not initialized")
        return
    
    try:
        with st.spinner("🧹 Clearing all documents..."):
            # Reset the vector store if available
            if hasattr(st.session_state.rag_system, 'get_all_document_ids'):
                # Use our newly added method to get all document IDs
                doc_ids = st.session_state.rag_system.get_all_document_ids()
                if doc_ids:
                    st.session_state.rag_system.vector_store.delete_documents(doc_ids)
                    st.info(f"Cleared {len(doc_ids)} documents from vector store")
            elif hasattr(st.session_state.rag_system, 'vector_store') and st.session_state.rag_system.vector_store:
                # Get all document IDs from vector store
                try:
                    # First try to use the get_all_documents method if available
                    if hasattr(st.session_state.rag_system.vector_store, 'get_all_documents'):
                        all_documents = st.session_state.rag_system.vector_store.get_all_documents()
                        if all_documents:
                            doc_ids = [doc.doc_id for doc in all_documents]
                            st.session_state.rag_system.vector_store.delete_documents(doc_ids)
                            st.info(f"Cleared {len(doc_ids)} documents from vector store")
                    # Fallback to using the ChromaDB collection directly
                    elif hasattr(st.session_state.rag_system.vector_store, 'collection'):
                        # Get all documents from the collection
                        all_docs = st.session_state.rag_system.vector_store.collection.get(
                            include=["documents", "metadatas", "embeddings"],
                            limit=10000  # Set a reasonable limit
                        )
                        
                        # If there are documents, delete them
                        if all_docs and 'ids' in all_docs and all_docs['ids']:
                            st.session_state.rag_system.vector_store.delete_documents(all_docs['ids'])
                            st.info(f"Cleared {len(all_docs['ids'])} documents from vector store")
                except Exception as e:
                    st.warning(f"Error clearing vector store: {str(e)}")
                            
            # Reset hybrid retriever index if it exists
            if (hasattr(st.session_state.rag_system, 'hybrid_retriever') and 
                hasattr(st.session_state.rag_system.hybrid_retriever, 'keyword_searcher')):
                # Reset the keyword searcher
                st.session_state.rag_system.hybrid_retriever.keyword_searcher.documents = []
                st.session_state.rag_system.hybrid_retriever.keyword_searcher.doc_frequencies = {}
                st.session_state.rag_system.hybrid_retriever.keyword_searcher.doc_lengths = []
                st.session_state.rag_system.hybrid_retriever.keyword_searcher.avg_doc_length = 0
                st.session_state.rag_system.hybrid_retriever.documents_indexed = False
                st.info("Reset keyword search index")
                
            # Reset the document count
            st.session_state.document_count = 0
            
            # Re-initialize to ensure we get a clean system
            st.session_state.system_initialized = False
            clean_domain = clean_domain_name(st.session_state.current_domain)
            st.session_state.rag_system = RAGExpertSystem(domain=clean_domain)
            initialization_result = st.session_state.rag_system.initialize()
            
            if initialization_result:
                st.session_state.system_initialized = True
                st.success("✅ All documents cleared successfully!")
            else:
                st.error("❌ Failed to re-initialize the system after clearing documents")
            # Reset the document count
            update_document_count()
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error clearing documents: {str(e)}")

def adjust_color_brightness(hex_color, percent):
    """Adjust the brightness of a hex color
    
    Args:
        hex_color: The hex color to adjust (e.g., "#FF5733")
        percent: The percentage to adjust by (positive brightens, negative darkens)
        
    Returns:
        Adjusted hex color
    """
    # Handle missing # - assume hex_color is already in hex format
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    
    # Handle short hex colors like #F00
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    
    # Try/except to handle any parsing errors
    try:
        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16) if len(hex_color) >= 6 else 0
        
        # Adjust brightness
        r = min(max(r + (r * percent / 100), 0), 255)
        g = min(max(g + (g * percent / 100), 0), 255)
        b = min(max(b + (b * percent / 100), 0), 255)
        
        # Convert back to hex
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    except Exception:
        # Return original color if any error occurs
        return f"#{hex_color}"

def get_file_icon(filename):
    """Get appropriate icon for a file based on its extension"""
    if not filename:
        return "📄"  # Default document icon
        
    ext = filename.lower().split('.')[-1] if '.' in filename else ""
    
    # Map file extensions to icons
    icons = {
        # Documents
        'pdf': "📕", 
        'txt': "📝", 
        'docx': "📄", 
        'doc': "📄",
        'rtf': "📝", 
        'odt': "📝",
        
        # Spreadsheets
        'xlsx': "📊", 
        'xls': "📊", 
        'csv': "📊", 
        'ods': "📊",
        
        # Presentations
        'pptx': "🖼️", 
        'ppt': "🖼️", 
        'odp': "🖼️",
        
        # Web & Markup
        'html': "🌐", 
        'htm': "🌐", 
        'xml': "🌐", 
        'md': "📝", 
        'rst': "📝",
        
        # Data formats
        'json': "📦", 
        'yaml': "📦", 
        'yml': "📦", 
        'toml': "📦", 
        'ini': "📦",
        
        # Programming
        'py': "💻", 
        'js': "💻", 
        'ts': "💻", 
        'java': "💻", 
        'cpp': "💻", 
        'c': "💻", 
        'h': "💻", 
        'css': "💻", 
        'sql': "💻",
        
        # E-books
        'epub': "📚", 
        'tex': "📚", 
        'bib': "📚",
        
        # Archives
        'zip': "🗃️", 
        'tar': "🗃️", 
        'gz': "🗃️"
    }
    
    return icons.get(ext, "📄")  # Return the icon or default to document

def direct_add_processed_document(processed_doc, file_path):
    """
    Directly add a processed document to the system, bypassing the standard pipeline
    for cases where we need more control over content extraction
    """
    try:
        # Ensure the processed_doc object is valid
        if not processed_doc or not hasattr(processed_doc, 'content') or not processed_doc.content:
            logger.error(f"Invalid processed document for {file_path}")
            return False
            
        # Create base metadata dict with safe fallbacks
        metadata = {
            "source_file": getattr(processed_doc, 'source_path', file_path) or file_path,
            "doc_type": getattr(processed_doc, 'doc_type', "unknown") or "unknown"
        }
        
        # Add additional metadata from processed_doc if it exists
        doc_metadata = getattr(processed_doc, 'metadata', None)
        if doc_metadata is not None:
            if isinstance(doc_metadata, dict):
                # Copy each key-value pair to avoid reference issues
                for k, v in doc_metadata.items():
                    # Convert list values to ChromaDB-compatible strings
                    if isinstance(v, list):
                        metadata[k] = ",".join(map(str, v)) if v else ""
                    elif isinstance(v, dict):
                        metadata[k] = str(v)
                    elif isinstance(v, (str, int, float, bool, type(None))):
                        metadata[k] = v
                    else:
                        metadata[k] = str(v)
            elif isinstance(doc_metadata, list):
                # If metadata is a list, convert to comma-separated string
                metadata["metadata_list"] = ",".join(map(str, doc_metadata)) if doc_metadata else ""
            else:
                # For any other type, convert to string and store
                metadata["original_metadata"] = str(doc_metadata)
        
        # Process the document using the chunker
        chunks = st.session_state.rag_system.chunker.chunk(
            processed_doc.content,
            metadata
        )
        
        if not chunks:
            logger.warning(f"No chunks created from document: {file_path}")
            return False
            
        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = st.session_state.rag_system.embedding_provider.embed_texts(chunk_texts)
        
        # Create documents for vector store and keyword docs for hybrid search
        documents = []
        keyword_docs = []
        
        for chunk, embedding in zip(chunks, embeddings):
            # Ensure chunk metadata is a dictionary with safe fallback
            chunk_metadata = {}
            
            # Copy metadata from chunk safely if it exists
            if hasattr(chunk, 'metadata'):
                if chunk.metadata is None:
                    # Leave chunk_metadata as empty dict
                    pass
                elif isinstance(chunk.metadata, dict):
                    # Make a clean copy of the metadata dict
                    for k, v in chunk.metadata.items():
                        chunk_metadata[k] = v
                else:
                    # For non-dict metadata, store it in a special key
                    chunk_metadata = {"original_metadata": str(chunk.metadata)}
                    logger.debug(f"Converted non-dict chunk metadata to dict in direct_add_processed_document: {type(chunk.metadata).__name__}")
            
            # Create document with safe metadata
            import time
            current_time = time.time()
            
            doc = Document(
                content=chunk.content,
                embedding=embedding,
                metadata=chunk_metadata,
                doc_id=chunk.chunk_id,
                timestamp=current_time
            )
            documents.append(doc)
            
            # Create keyword document entry with the same metadata
            keyword_docs.append({
                'content': chunk.content,
                'metadata': chunk_metadata,  # Use the same clean metadata
                'doc_id': chunk.chunk_id
            })
        logger.info(f"Prepared {len(documents)} documents and {len(keyword_docs)} keyword docs for {file_path}")
        
        # Add documents to vector store
        if documents:
            doc_ids = st.session_state.rag_system.vector_store.add_documents(documents)
            logger.info(f"Added {len(doc_ids)} chunks directly to vector store")
            
            # Index documents for hybrid search
            try:
                if hasattr(st.session_state.rag_system.hybrid_retriever, 'index_documents') and keyword_docs:
                    st.session_state.rag_system.hybrid_retriever.index_documents(keyword_docs)
                    st.session_state.rag_system.hybrid_retriever.documents_indexed = True
                    logger.info(f"Added {len(keyword_docs)} documents to hybrid index")
            except Exception as e:
                logger.warning(f"Error indexing documents for hybrid search: {e}")
                # We'll still return True even if hybrid indexing fails, as the documents are in the vector store
            
            return True
        return False
    
    except Exception as e:
        logger.error(f"Error directly adding document: {e}")
        return False

def update_document_count():
    """Update the document count in session state with actual count from vector store"""
    try:
        if st.session_state.system_initialized and hasattr(st.session_state.rag_system, 'vector_store'):
            # Get actual count from the vector store
            if hasattr(st.session_state.rag_system.vector_store, 'collection'):
                # Use ChromaDB's collection count method
                count_info = st.session_state.rag_system.vector_store.collection.count()
                st.session_state.document_count = count_info
                logger.info(f"Updated document count to {count_info} from ChromaDB collection")
            elif hasattr(st.session_state.rag_system.vector_store, 'get_all_documents'):
                # Fallback to get_all_documents method
                all_documents = st.session_state.rag_system.vector_store.get_all_documents()
                st.session_state.document_count = len(all_documents) if all_documents else 0
                logger.info(f"Updated document count to {st.session_state.document_count} from vector store")
    except Exception as e:
        logger.warning(f"Error updating document count: {e}")
        # If we can't get the count, don't change it

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    create_header()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/nolan/64/artificial-intelligence.png", width=64)
        st.markdown("# RAG Expert System")
        st.markdown("---")
        
        # Domain selector
        create_domain_selector()
        
        if st.session_state.current_domain:
            initialize_rag_system(st.session_state.current_domain)
        
        # Show system status if initialized
        if st.session_state.system_initialized:
            st.markdown("---")
            create_system_status()
            
            # Add logout button
            st.markdown("---")
            if st.button("🔄 Reset System", use_container_width=True):
                # Reset session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Main content area
    if not st.session_state.current_domain:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to the RAG Expert System
        
        This system allows you to upload documents and chat with an AI assistant about their contents.
        
        To get started:
        1. Select a domain from the sidebar
        2. Upload your documents
        3. Start asking questions
        
        Your documents stay private and are processed locally.
        """)
        
        # Feature showcase
        st.markdown("---")
        st.markdown("## ✨ Features")
        
        feature_cols = st.columns(3)
        with feature_cols[0]:
            st.markdown("""
            ### 📝 Document Processing
            - PDF, DOCX, TXT support
            - Automatic chunking
            - Semantic indexing
            """)
        
        with feature_cols[1]:
            st.markdown("""
            ### 🧠 Intelligent Querying
            - Natural language questions
            - Source citations
            - Advanced retrieval
            """)
        
        with feature_cols[2]:
            st.markdown("""
            ### 🛠️ Customization
            - Domain specialization
            - Multiple file formats
            - Privacy focused
            """)
    
    elif not st.session_state.system_initialized:
        st.warning("⚙️ System initialization in progress...")
    
    else:
        # Main application layout
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Documents", "📊 Analytics"])
        
        with tab1:
            create_chat_interface()
        
        with tab2:
            create_file_upload()
        
        with tab3:
            create_analytics()


if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="RAG Expert System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS for enhanced visual elements
    apply_custom_css()
    
    # Add global CSS for proper HTML rendering
    st.markdown("""
    <style>
    /* Fix for HTML rendering issues */
    .stMarkdown div {
        overflow: visible !important;
    }
    
    /* Ensure animations work properly */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        0% { opacity: 0; transform: translateX(-10px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .status-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .status-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .status-number {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
            animation: pulse 2s infinite;
        }
        .status-label {
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        .privacy-badge {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 1rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .source-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        .source-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #764ba2;
        }
        .source-header {
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        .source-icon {
            font-size: 1.2rem;
            margin-right: 0.5rem;
        }
        .source-title {
            font-weight: 600;
            color: #495057;
            margin: 0;
        }
        .source-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
            font-size: 0.85rem;
        }
        .source-meta-item {
            background: #fff;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            color: #6c757d;
        }
        .source-url {
            color: #0d6efd;
            text-decoration: underline;
        }
        .source-preview {
            background: #fff;
            padding: 0.5rem;
            border-radius: 5px;
            margin-top: 0.5rem;
            font-size: 0.85em;
            color: #495057;
            border-left: 3px solid #28a745;
            max-height: 100px;
            overflow-y: auto;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .chat-message {
            animation: fadeIn 0.5s ease-out;
        }
        .domain-button {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .domain-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    main()
