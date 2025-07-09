"""
Enhanced FastAPI backend for RAG Expert System with document prioritization
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import tempfile
import os
import uuid
from datetime import datetime

from src.rag_system import RAGExpertSystem
from src.llm_providers import GenerationConfig
from src.source_formatter import format_sources

class QueryRequest(BaseModel):
    question: str
    max_chunks: Optional[int] = 5
    retrieval_method: Optional[str] = "hybrid"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    prioritize_uploads: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    model_used: str
    timestamp: str
    retrieval_method: str
    processing_time: float
    session_id: str

class SystemStats(BaseModel):
    domain: str
    session_id: str
    is_initialized: bool
    indexed_documents: int
    total_queries: int
    embedding_model: Optional[str]
    vector_store_type: str
    llm_model: Optional[str]
    supported_formats: List[str]
    user_uploaded_docs: List[str]

class InitializeRequest(BaseModel):
    domain: Optional[str] = "general"
    session_id: Optional[str] = None
    use_separate_collection_for_uploads: Optional[bool] = True

# Track user uploaded documents separately
class SessionData:
    def __init__(self, rag_system: RAGExpertSystem):
        self.rag_system = rag_system
        self.uploaded_docs = []  # List of document paths uploaded by user
        self.upload_timestamp = None  # When the latest document was uploaded

# Map of session_id to session data
sessions: Dict[str, SessionData] = {}

app = FastAPI(
    title="RAG Expert System API",
    description="Advanced Retrieval-Augmented Generation System with document prioritization",
    version="1.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/initialize", response_model=dict)
async def initialize_system(request: InitializeRequest):
    """Initialize a new RAG system instance"""
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id in sessions:
        return {
            "message": "System already initialized",
            "session_id": session_id,
            "domain": sessions[session_id].rag_system.domain
        }
    
    try:
        rag_system = RAGExpertSystem(domain=request.domain, session_id=session_id)
        
        if rag_system.initialize():
            # Create session data to track user uploads
            sessions[session_id] = SessionData(rag_system)
            
            return {
                "message": "RAG system initialized successfully",
                "session_id": session_id,
                "domain": request.domain
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize RAG system")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing system: {str(e)}")


@app.post("/upload-documents/{session_id}")
async def upload_documents(
    session_id: str,
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None),
    create_separate_collection: Optional[bool] = Form(True)
):
    """Upload and process documents"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Initialize system first.")
    
    session_data = sessions[session_id]
    rag_system = session_data.rag_system
    
    try:
        # Save uploaded files temporarily
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            file_paths.append(file_path)
        
        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            try:
                import json
                doc_metadata = json.loads(metadata)
            except:
                pass
        
        # Add custom metadata to identify user-uploaded documents
        user_metadata = {
            **doc_metadata,
            "is_user_uploaded": True,
            "upload_timestamp": datetime.now().isoformat()
        }
        
        # Process documents
        success = rag_system.add_documents(file_paths, user_metadata)
        
        if success:
            # Track uploaded documents in session data
            session_data.uploaded_docs.extend([os.path.basename(path) for path in file_paths])
            session_data.upload_timestamp = datetime.now()
            
            # Cleanup temporary files
            for file_path in file_paths:
                try:
                    os.remove(file_path)
                except:
                    pass
            
            return {
                "message": f"Successfully processed {len(file_paths)} documents",
                "files_processed": [file.filename for file in files],
                "exclusive_mode_activated": True,
                "note": "Answers will now be sourced ONLY from your uploaded documents"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to process some documents")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")


@app.post("/add-directory/{session_id}")
async def add_directory(
    session_id: str,
    directory_path: str = Form(...),
    recursive: bool = Form(True),
    metadata: Optional[str] = Form(None)
):
    """Add documents from a directory"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Initialize system first.")
    
    session_data = sessions[session_id]
    rag_system = session_data.rag_system
    
    try:
        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            try:
                import json
                doc_metadata = json.loads(metadata)
            except:
                pass
        
        # Add custom metadata to identify user-added documents
        user_metadata = {
            **doc_metadata,
            "is_user_uploaded": True,
            "upload_timestamp": datetime.now().isoformat()
        }
        
        success = rag_system.add_directory(directory_path, recursive, user_metadata)
        
        if success:
            # Update session data
            session_data.upload_timestamp = datetime.now()
            
            # Get processed files
            import glob
            supported_formats = rag_system.document_processor.get_supported_formats()
            pattern = os.path.join(directory_path, "**" if recursive else "*")
            
            for ext in supported_formats:
                matching_files = glob.glob(f"{pattern}*{ext}", recursive=recursive)
                session_data.uploaded_docs.extend([os.path.basename(path) for path in matching_files])
            
            return {
                "message": f"Successfully processed directory: {directory_path}",
                "recursive": recursive,
                "exclusive_mode_activated": True,
                "note": "Answers will now be sourced ONLY from your uploaded documents"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to process directory")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing directory: {str(e)}")


@app.post("/query/{session_id}", response_model=QueryResponse)
async def query_system(
    session_id: str, 
    request: QueryRequest,
    prioritize_uploads: bool = Query(True, description="Use only user uploaded documents when available")
):
    """Query the RAG system - uses ONLY user uploaded documents if available"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Initialize system first.")
    
    session_data = sessions[session_id]
    rag_system = session_data.rag_system
    
    try:
        generation_config = GenerationConfig(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # If user has uploaded documents, use ONLY those documents
        custom_search_params = {}
        exclusive_mode = False
        
        if session_data.uploaded_docs and (prioritize_uploads or request.prioritize_uploads):
            # Create a filter to use ONLY user-uploaded documents
            custom_search_params["filter"] = {"is_user_uploaded": True}
            exclusive_mode = True
        
        # Add relevance threshold to prioritize higher quality matches
        # Default relevance threshold is higher for the exclusive mode to ensure quality
        relevance_threshold = 0.3 if exclusive_mode else 0.1
        custom_search_params["relevance_threshold"] = relevance_threshold
        
        # Now we use the custom_search_params with our updated RAG system
        response = rag_system.query(
            request.question,
            max_chunks=request.max_chunks,
            generation_config=generation_config,
            retrieval_method=request.retrieval_method,
            custom_search_params=custom_search_params
        )
        
        # Add indicator if the response used only uploaded documents
        if exclusive_mode:
            response.answer = f"{response.answer}\n\n[Response based EXCLUSIVELY on your uploaded documents. Default knowledge base was NOT used.]"
        else:
            # No user uploads, so we're using the default knowledge base
            response.answer = f"{response.answer}\n\n[Response from default knowledge base. Upload documents to get answers from your own content.]"
        
        # Enhance source information with formatted details
        enhanced_sources = format_sources(response.sources)
        
        return QueryResponse(
            answer=response.answer,
            sources=enhanced_sources,  # Use enhanced source information
            query=response.query,
            model_used=response.model_used,
            timestamp=response.timestamp.isoformat(),
            retrieval_method=response.retrieval_method,
            processing_time=response.processing_time,
            session_id=response.session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/explain-retrieval/{session_id}")
async def explain_retrieval(
    session_id: str, 
    question: str, 
    max_chunks: int = 5,
    prioritize_uploads: bool = Query(True, description="Use only user uploaded documents when available")
):
    """Explain how retrieval works - uses ONLY user uploaded documents if available"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Initialize system first.")
    
    session_data = sessions[session_id]
    rag_system = session_data.rag_system
    
    try:
        # Add filter to use ONLY uploaded documents if available
        custom_search_params = {}
        exclusive_mode = False
        
        if session_data.uploaded_docs and prioritize_uploads:
            custom_search_params["filter"] = {"is_user_uploaded": True}
            exclusive_mode = True
        
        # Add relevance threshold to prioritize higher quality matches
        # Default relevance threshold is higher for the exclusive mode to ensure quality
        relevance_threshold = 0.3 if exclusive_mode else 0.1
        custom_search_params["relevance_threshold"] = relevance_threshold
            
        # Now we use the custom_search_params with our updated RAG system
        explanation = rag_system.explain_retrieval(question, max_chunks, custom_search_params)
        
        # Convert SearchResult objects to dictionaries with enhanced information
        def convert_search_results(results):
                    # Convert to standard format first
            basic_results = [
                {
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata,
                    "source": result.source,
                    "doc_id": result.doc_id,
                    "is_user_uploaded": result.metadata.get("is_user_uploaded", False) 
                    if result.metadata else False
                }
                for result in results
            ]
            
            # Apply source formatting enhancements
            return format_sources(basic_results)
            return format_sources(basic_results)
            return format_sources(basic_results)
        
        return {
            "query": explanation["query"],
            "vector_results": convert_search_results(explanation["vector_results"]),
            "keyword_results": convert_search_results(explanation["keyword_results"]),
            "hybrid_results": convert_search_results(explanation["hybrid_results"]),
            "alpha": explanation["alpha"],
            "total_documents": explanation["total_documents"],
            "exclusive_mode": exclusive_mode,
            "using_only_uploads": exclusive_mode,
            "uploads_available": bool(session_data.uploaded_docs)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error explaining retrieval: {str(e)}")


@app.get("/stats/{session_id}", response_model=SystemStats)
async def get_system_stats(session_id: str):
    """Get system statistics"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Initialize system first.")
    
    session_data = sessions[session_id]
    rag_system = session_data.rag_system
    stats = rag_system.get_system_stats()
    
    # Add info about user uploaded documents
    stats["user_uploaded_docs"] = session_data.uploaded_docs
    
    return SystemStats(**stats)


@app.get("/audit-logs/{session_id}")
async def get_audit_logs(session_id: str):
    """Get audit logs for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Initialize system first.")
    
    session_data = sessions[session_id]
    rag_system = session_data.rag_system
    
    logs = []
    for log in rag_system.audit_logs:
        logs.append({
            "session_id": log.session_id,
            "query": log.query,
            "response": log.response,
            "sources_used": log.sources_used,
            "timestamp": log.timestamp.isoformat(),
            "model_used": log.model_used,
            "retrieval_method": log.retrieval_method,
            "processing_time": log.processing_time,
            "user_feedback": log.user_feedback
        })
    
    return {"logs": logs, "total_queries": len(logs)}


@app.post("/export-logs/{session_id}")
async def export_audit_logs(session_id: str):
    """Export audit logs to file"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Initialize system first.")
    
    session_data = sessions[session_id]
    rag_system = session_data.rag_system
    
    try:
        file_path = rag_system.export_audit_logs()
        return {
            "message": "Audit logs exported successfully",
            "file_path": file_path
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting logs: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and cleanup resources"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    
    return {"message": f"Session {session_id} deleted successfully"}


@app.get("/toggle-exclusive-mode/{session_id}")
async def toggle_exclusive_mode(
    session_id: str,
    enable: bool = Query(True, description="Enable or disable exclusive use of uploaded documents")
):
    """Toggle exclusive use of user-uploaded documents (ignore knowledge base when uploads exist)"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Initialize system first.")
    
    session_data = sessions[session_id]
    
    if not session_data.uploaded_docs:
        return {
            "message": "No uploaded documents found for this session",
            "exclusive_mode": False
        }
    
    return {
        "message": f"Exclusive document mode {'enabled' if enable else 'disabled'}",
        "exclusive_mode": enable,
        "uploaded_documents": len(session_data.uploaded_docs),
        "note": "When enabled, only uploaded documents will be used for answers (knowledge base ignored)"
    }


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    sessions_info = []
    for session_id, session_data in sessions.items():
        rag_system = session_data.rag_system
        stats = rag_system.get_system_stats()
        sessions_info.append({
            "session_id": session_id,
            "domain": stats["domain"],
            "indexed_documents": stats["indexed_documents"],
            "user_uploaded_docs": len(session_data.uploaded_docs),
            "total_queries": stats["total_queries"],
            "is_initialized": stats["is_initialized"]
        })
    
    return {"sessions": sessions_info, "total_sessions": len(sessions)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(sessions)
    }


@app.get("/document-mode-info/{session_id}")
async def document_mode_info(session_id: str):
    """Get information about how documents are being used for answers"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Initialize system first.")
    
    session_data = sessions[session_id]
    
    if not session_data.uploaded_docs:
        return {
            "mode": "default_knowledge_base",
            "description": "Using default knowledge base only (no uploaded documents)",
            "uploaded_documents": 0,
            "note": "Upload documents to activate exclusive mode"
        }
    
    return {
        "mode": "exclusive_uploaded_documents",
        "description": "Using ONLY your uploaded documents for answers",
        "uploaded_documents": len(session_data.uploaded_docs),
        "documents": session_data.uploaded_docs,
        "note": "The system will ignore the default knowledge base and answer exclusively from your documents"
    }


@app.post("/generate-questions/{session_id}")
async def generate_questions(
    session_id: str,
    num_questions: int = Query(10, description="Number of questions to generate"),
    difficulty: str = Query("difficult", description="Question difficulty (easy, medium, difficult)"),
    topic: Optional[str] = Query(None, description="Specific topic to focus on"),
    use_uploads_only: bool = Query(True, description="Use only user uploaded documents")
):
    """Generate questions from documents with specified difficulty"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Initialize system first.")
    
    session_data = sessions[session_id]
    rag_system = session_data.rag_system
    
    try:
        # Check if we have documents to generate questions from
        if use_uploads_only and not session_data.uploaded_docs:
            return {
                "message": "No uploaded documents found for question generation",
                "questions": [],
                "success": False
            }
        
        # Create a prompt for question generation
        custom_search_params = {}
        if use_uploads_only and session_data.uploaded_docs:
            custom_search_params["filter"] = {"is_user_uploaded": True}
        
        generation_config = GenerationConfig(
            temperature=0.8,  # Higher temperature for more creative questions
            max_tokens=2000
        )
        
        # First, generate a query to retrieve relevant documents
        retrieval_query = f"Extract comprehensive information about {topic or 'all important topics'}"
        
        # Retrieve context information from documents
        response = rag_system.query(
            retrieval_query,
            max_chunks=10,  # Get more chunks for better coverage
            generation_config=generation_config,
            custom_search_params=custom_search_params
        )
        
        # Now generate questions based on the retrieved information
        question_gen_prompt = f"""Based on the following information, generate {num_questions} {difficulty} questions. 
For each question, include the correct answer and explanation.

INFORMATION:
{response.answer}

FORMAT:
1. [Question]
   - Answer: [Correct answer]
   - Explanation: [Brief explanation of why this is correct]

Make sure the questions are {difficulty} level and test deep understanding of the concepts.
"""
        
        # Use the RAG system for question generation
        question_response = rag_system.query(
            question_gen_prompt,
            generation_config=GenerationConfig(
                temperature=0.9,
                max_tokens=2500
            )
        )
        
        # Combine and format sources from both responses
        all_sources = response.sources + question_response.sources
        enhanced_sources = format_sources(all_sources)
        
        return {
            "message": f"Successfully generated {num_questions} {difficulty} questions",
            "questions": question_response.answer,
            "sources": enhanced_sources,  # Use enhanced sources
            "success": True,
            "using_uploads_only": use_uploads_only and bool(session_data.uploaded_docs)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
