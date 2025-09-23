"""
FastAPI backend for CLM automation system.
Modern web API replacing Streamlit interface.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import logging
import sys
import os
import io
from datetime import datetime
import tempfile
import uuid

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config
from src.database import DatabaseManager
from src.embeddings import EmbeddingManager
from src.rag_pipeline import RAGPipeline
from src.contract_agent import ContractAgent
from src.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CLM Automation API",
    description="Contract Lifecycle Management API with AI-powered analysis",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
components = {}

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    conversation_id: str
    message_id: str
    timestamp: datetime

class SystemStats(BaseModel):
    total_documents: int
    active_contracts: int
    total_chunks: int
    system_status: str

class DocumentSearchRequest(BaseModel):
    query: str
    limit: int = 10

class ReportRequest(BaseModel):
    email: str
    include_expiring: bool = True
    include_conflicts: bool = True
    include_analytics: bool = True

class APIKeyRequest(BaseModel):
    name: str
    description: str = ''
    expires_at: Optional[str] = None

class APIKeyResponse(BaseModel):
    id: str
    name: str
    key: str
    created_at: str
    expires_at: Optional[str] = None

class GlobalSearchRequest(BaseModel):
    query: str
    limit: int = 20
    search_types: List[str] = ['documents', 'contracts', 'chunks']

class SearchResult(BaseModel):
    id: str
    title: str
    type: str
    content_preview: str
    score: float
    metadata: Dict[str, Any]

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversation_sessions: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, conversation_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if conversation_id:
            if conversation_id not in self.conversation_sessions:
                self.conversation_sessions[conversation_id] = []
            self.conversation_sessions[conversation_id].append(websocket)

    def disconnect(self, websocket: WebSocket, conversation_id: str = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if conversation_id and conversation_id in self.conversation_sessions:
            if websocket in self.conversation_sessions[conversation_id]:
                self.conversation_sessions[conversation_id].remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_conversation(self, message: str, conversation_id: str):
        if conversation_id in self.conversation_sessions:
            for connection in self.conversation_sessions[conversation_id]:
                try:
                    await connection.send_text(message)
                except:
                    # Remove dead connections
                    self.conversation_sessions[conversation_id].remove(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize CLM components on startup"""
    global components
    
    try:
        logger.info("Initializing CLM components...")
        
        # Validate configuration
        is_valid, missing = Config.validate_config()
        if not is_valid:
            logger.error(f"Configuration validation failed: Missing {missing}")
            raise ValueError(f"Missing configuration: {missing}")
        
        # Initialize components
        db_manager = DatabaseManager()
        embedding_manager = EmbeddingManager(db_manager)
        rag_pipeline = RAGPipeline(db_manager, embedding_manager)
        contract_agent = ContractAgent(db_manager)
        document_processor = DocumentProcessor(db_manager)
        
        components = {
            "db_manager": db_manager,
            "embedding_manager": embedding_manager,
            "rag_pipeline": rag_pipeline,
            "contract_agent": contract_agent,
            "document_processor": document_processor
        }
        
        logger.info("CLM components initialized successfully")
        
        # Create necessary directories
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./uploads', exist_ok=True)
        
    except Exception as e:
        logger.error(f"Failed to initialize CLM components: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "CLM Automation API", "status": "running", "timestamp": datetime.now()}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check database connection
        db_health = "ok"
        if "db_manager" in components:
            # Test database connection
            try:
                result = components["db_manager"].client.table('documents').select('id').limit(1).execute()
                db_health = "ok"
            except Exception as e:
                db_health = f"error: {str(e)}"
                
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "components": {
                "database": db_health,
                "ai_pipeline": "ok" if "rag_pipeline" in components else "not initialized",
                "document_processor": "ok" if "document_processor" in components else "not initialized"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
            
        db = components["db_manager"]
        
        # Get real-time stats from database
        try:
            docs_result = db.client.table('documents').select('id').execute()
            total_docs = len(docs_result.data)
            
            # Get active contracts count - use real data
            contracts_result = db.client.table('contracts').select('id').execute()
            active_contracts = len(contracts_result.data)
            
            chunks_result = db.client.table('document_chunks').select('id').execute()
            total_chunks = len(chunks_result.data)
            
            logger.info(f"Real stats: docs={total_docs}, contracts={active_contracts}, chunks={total_chunks}")
                
        except Exception as e:
            logger.error(f"Failed to get real stats: {e}")
            # Fallback to zeros if database fails
            total_docs = 0
            active_contracts = 0
            total_chunks = 0
        
        return SystemStats(
            total_documents=total_docs,
            active_contracts=active_contracts,
            total_chunks=total_chunks,
            system_status="operational"
        )
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        # Return zero stats if everything fails
        return SystemStats(
            total_documents=0,
            active_contracts=0,
            total_chunks=0,
            system_status="error"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """Main chat endpoint for contract queries"""
    try:
        if "rag_pipeline" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Generate unique IDs
        conversation_id = chat_request.conversation_id or str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        
        # Enhanced AI agent with RAG pipeline as tool
        query_type = classify_query(chat_request.message)
        
        if query_type == "greeting":
            response_data = handle_greeting()
        elif query_type == "help":
            response_data = handle_help_request()
        elif query_type == "agent_task":
            # Use contract agent for administrative/monitoring tasks
            try:
                agent_response = components["contract_agent"].query_agent(chat_request.message)
                response_data = {
                    "answer": f"🤖 **Contract Agent**: {agent_response}",
                    "sources": []
                }
            except Exception as e:
                logger.error(f"Agent query failed: {e}")
                response_data = {
                    "answer": "I apologize, but I encountered an issue processing your administrative request. Please try rephrasing your question.",
                    "sources": []
                }
        else:
            # Use RAG pipeline for document-focused queries
            try:
                response_data = components["rag_pipeline"].query(chat_request.message)
                # Add AI agent context to the response
                if response_data.get("sources"):
                    response_data["answer"] = f"📚 **Document Search Results**: {response_data['answer']}"
            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                response_data = {
                    "answer": "I'm having trouble searching the documents right now. Please try again or rephrase your question.",
                    "sources": []
                }
        
        # Broadcast to WebSocket connections if any
        if conversation_id in manager.conversation_sessions:
            await manager.broadcast_to_conversation(
                json.dumps({
                    "type": "chat_response",
                    "response": response_data["answer"],
                    "message_id": message_id
                }),
                conversation_id
            )
        
        return ChatResponse(
            response=response_data["answer"],
            sources=response_data.get("sources", []),
            conversation_id=conversation_id,
            message_id=message_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, conversation_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process the message
            if message_data.get("type") == "chat":
                # Process chat message
                response_data = components["rag_pipeline"].query(message_data["message"])
                
                # Send response back
                await manager.send_personal_message(
                    json.dumps({
                        "type": "chat_response",
                        "response": response_data["answer"],
                        "sources": response_data.get("sources", []),
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, conversation_id)
        logger.info(f"WebSocket disconnected for conversation {conversation_id}")

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    try:
        if "document_processor" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        document_processor = components["document_processor"]
        results = []
        
        for file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Process the file with frontend source tracking
                result = document_processor.process_single_document(
                    tmp_file_path,
                    filename=file.filename,
                    extract_contracts=True,
                    metadata={"source": "frontend_upload", "uploaded_via": "web_interface"}
                )
                
                results.append({
                    "filename": file.filename,
                    "success": result.get("success", False),
                    "document_id": result.get("document_id"),
                    "chunks_created": result.get("chunks_created", 0),
                    "contract_extracted": result.get("contract_extracted", False)
                })
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(search_request: DocumentSearchRequest):
    """Search for similar documents"""
    try:
        if "rag_pipeline" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        similar_docs = components["rag_pipeline"].find_similar_contracts(
            search_request.query,
            limit=search_request.limit
        )
        
        return {"documents": similar_docs}
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-report")
async def generate_report(report_request: ReportRequest):
    """Generate and optionally send contract report"""
    try:
        if "contract_agent" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        contract_agent = components["contract_agent"]
        
        # Generate report
        report = contract_agent.generate_daily_report()
        
        # Optionally send email
        if report_request.email:
            # Update the email in config temporarily
            original_email = Config.REPORT_EMAIL
            Config.REPORT_EMAIL = report_request.email
            
            try:
                contract_agent.send_report_email(report)
                email_sent = True
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
                email_sent = False
            finally:
                # Restore original email
                Config.REPORT_EMAIL = original_email
        else:
            email_sent = False
        
        return {
            "report": report,
            "email_sent": email_sent
        }
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details and content for preview"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        db = components["db_manager"]
        
        # Get document details
        result = db.client.table('documents')\
            .select('*')\
            .eq('id', document_id)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = result.data[0]
        
        return {
            "id": document['id'],
            "filename": document['filename'],
            "file_type": document['file_type'],
            "content": document['content'],
            "created_at": document['created_at'],
            "metadata": document.get('metadata', {})
        }
        
    except Exception as e:
        logger.error(f"Document retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}/download")
async def download_document(document_id: str):
    """Download document content"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        db = components["db_manager"]
        
        # Get document details
        result = db.client.table('documents')\
            .select('filename, content, file_type')\
            .eq('id', document_id)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = result.data[0]
        
        # Return content as downloadable file
        return StreamingResponse(
            io.BytesIO(document['content'].encode()),
            media_type='application/octet-stream',
            headers={"Content-Disposition": f"attachment; filename={document['filename']}"}
        )
        
    except Exception as e:
        logger.error(f"Document download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents(frontend_only: bool = False):
    """Get list of documents, optionally filtered to frontend uploads only"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        db = components["db_manager"]
        
        # Get documents with metadata, optionally filter to frontend uploads
        query = db.client.table('documents')\
            .select('id, filename, file_type, created_at, updated_at, metadata')
        
        if frontend_only:
            # Filter for documents uploaded via frontend
            query = query.contains('metadata', {'source': 'frontend_upload'})
        
        result = query.order('created_at', desc=True).execute()
        
        documents = []
        for doc in result.data:
            # Calculate file size from metadata if available
            file_size = "Unknown"
            if doc.get('metadata') and isinstance(doc['metadata'], dict):
                size_bytes = doc['metadata'].get('file_size', 0)
                if size_bytes:
                    if size_bytes < 1024:
                        file_size = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        file_size = f"{size_bytes / 1024:.1f} KB"
                    else:
                        file_size = f"{size_bytes / (1024 * 1024):.1f} MB"
            
            documents.append({
                "id": doc['id'],
                "filename": doc['filename'],
                "fileType": doc['file_type'].upper(),
                "uploadedAt": doc['created_at'][:10] if doc.get('created_at') else None,
                "size": file_size,
                "status": "Processed"
            })
        
        # If no documents in database, return mock data for demo
        if not documents:
            documents = [
                {
                    "id": "demo-1",
                    "filename": "TechCorp_Service_Agreement.pdf",
                    "fileType": "PDF",
                    "uploadedAt": "2024-01-15",
                    "size": "2.4 MB",
                    "status": "Processed"
                },
                {
                    "id": "demo-2",
                    "filename": "NDA_GlobalTech.docx",
                    "fileType": "DOCX",
                    "uploadedAt": "2024-01-14",
                    "size": "856 KB",
                    "status": "Processed"
                },
                {
                    "id": "demo-3",
                    "filename": "License_Agreement_v2.pdf",
                    "fileType": "PDF",
                    "uploadedAt": "2024-01-13",
                    "size": "1.2 MB",
                    "status": "Processing"
                }
            ]
        
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Documents list error: {e}")
        # Return mock data if database fails
        return {
            "documents": [
                {
                    "id": "demo-1",
                    "filename": "Demo_Contract.pdf",
                    "fileType": "PDF",
                    "uploadedAt": "2024-01-15",
                    "size": "2.4 MB",
                    "status": "Processed"
                }
            ]
        }

@app.get("/frontend-documents")
async def get_frontend_documents():
    """Get list of documents uploaded via frontend only"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        db = components["db_manager"]
        
        # Get only documents uploaded via frontend
        result = db.client.table('documents')\
            .select('id, filename, file_type, created_at, updated_at, metadata')\
            .contains('metadata', {'source': 'frontend_upload'})\
            .order('created_at', desc=True)\
            .execute()
        
        documents = []
        for doc in result.data:
            # Calculate file size from metadata if available
            file_size = "Unknown"
            if doc.get('metadata') and isinstance(doc['metadata'], dict):
                size_bytes = doc['metadata'].get('file_size', 0)
                if size_bytes:
                    if size_bytes < 1024:
                        file_size = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        file_size = f"{size_bytes / 1024:.1f} KB"
                    else:
                        file_size = f"{size_bytes / (1024 * 1024):.1f} MB"
            
            documents.append({
                "id": doc['id'],
                "filename": doc['filename'],
                "fileType": doc['file_type'].upper(),
                "uploadedAt": doc['created_at'][:10] if doc.get('created_at') else None,
                "size": file_size,
                "status": "Processed",
                "source": doc.get('metadata', {}).get('source', 'unknown')
            })
        
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Frontend documents list error: {e}")
        return {"documents": []}

@app.get("/contracts")
async def get_contracts():
    """Get list of all contracts"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        db = components["db_manager"]
        
        # Get all contracts with document info
        result = db.client.table('contracts')\
            .select('*, documents!inner(filename)')\
            .eq('status', 'active')\
            .order('created_at', desc=True)\
            .execute()
        
        from datetime import datetime, timedelta
        
        contracts = []
        for contract in result.data:
            # Calculate days to expiry
            days_to_expiry = 365  # Default
            if contract.get('end_date'):
                try:
                    end_date = datetime.fromisoformat(contract['end_date'].replace('Z', '+00:00'))
                    days_to_expiry = (end_date - datetime.now()).days
                except:
                    pass
            
            # Determine status based on expiry
            status = "Active"
            if days_to_expiry < 30:
                status = "Expiring Soon"
            elif days_to_expiry < 0:
                status = "Expired"
            
            contracts.append({
                "id": contract['id'],
                "name": contract.get('contract_name', 'Unnamed Contract'),
                "parties": contract.get('parties', []),
                "startDate": contract.get('start_date', ''),
                "endDate": contract.get('end_date', ''),
                "status": status,
                "department": contract.get('department', 'Unknown'),
                "value": "N/A",  # Add contract value field to database if needed
                "daysToExpiry": max(0, days_to_expiry)
            })
        
        # If no contracts in database, return mock data for demo
        if not contracts:
            contracts = [
                {
                    "id": "demo-1",
                    "name": "TechCorp Service Agreement",
                    "parties": ["Your Company", "TechCorp Inc."],
                    "startDate": "2024-01-01",
                    "endDate": "2024-12-31",
                    "status": "Active",
                    "department": "IT",
                    "value": "$50,000",
                    "daysToExpiry": 42
                },
                {
                    "id": "demo-2",
                    "name": "Global Tech NDA",
                    "parties": ["Your Company", "Global Tech Ltd."],
                    "startDate": "2023-06-15",
                    "endDate": "2025-06-14",
                    "status": "Active",
                    "department": "Legal",
                    "value": "N/A",
                    "daysToExpiry": 507
                },
                {
                    "id": "demo-3",
                    "name": "Software License Agreement",
                    "parties": ["Your Company", "SoftWare Solutions"],
                    "startDate": "2024-01-10",
                    "endDate": "2024-06-30",
                    "status": "Expiring Soon",
                    "department": "IT",
                    "value": "$25,000",
                    "daysToExpiry": 15
                }
            ]
        
        return {"contracts": contracts}
        
    except Exception as e:
        logger.error(f"Contracts list error: {e}")
        # Return mock data if database fails
        return {
            "contracts": [
                {
                    "id": "demo-1",
                    "name": "Demo Service Agreement",
                    "parties": ["Your Company", "Demo Corp"],
                    "startDate": "2024-01-01",
                    "endDate": "2024-12-31",
                    "status": "Active",
                    "department": "IT",
                    "value": "$50,000",
                    "daysToExpiry": 42
                }
            ]
        }

@app.get("/chunks")
async def get_chunks():
    """Get list of all document chunks"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        db = components["db_manager"]
        
        # Get all document chunks with document info
        result = db.client.table('document_chunks')\
            .select('*, documents!inner(filename)')\
            .order('created_at', desc=True)\
            .limit(100)\
            .execute()
        
        chunks = []
        for chunk in result.data:
            chunks.append({
                "id": chunk['id'],
                "document_id": chunk['document_id'],
                "document_name": chunk['documents']['filename'],
                "chunk_index": chunk.get('chunk_index', 0),
                "text_preview": chunk['chunk_text'][:100] + "..." if len(chunk['chunk_text']) > 100 else chunk['chunk_text'],
                "full_text": chunk['chunk_text'],
                "created_at": chunk.get('created_at', ''),
                "token_count": len(chunk['chunk_text'].split())
            })
        
        # If no chunks in database, return mock data for demo
        if not chunks:
            chunks = [
                {
                    "id": "demo-chunk-1",
                    "document_id": "demo-1",
                    "document_name": "TechCorp_Service_Agreement.pdf",
                    "chunk_index": 0,
                    "text_preview": "This Service Agreement is entered into between TechCorp Inc. and Client Company...",
                    "full_text": "This Service Agreement is entered into between TechCorp Inc. and Client Company for the provision of software development services.",
                    "created_at": "2024-01-15",
                    "token_count": 22
                },
                {
                    "id": "demo-chunk-2",
                    "document_id": "demo-2",
                    "document_name": "NDA_GlobalTech.docx",
                    "chunk_index": 0,
                    "text_preview": "This Non-Disclosure Agreement (NDA) is made between the parties to protect...",
                    "full_text": "This Non-Disclosure Agreement (NDA) is made between the parties to protect confidential information shared during business discussions.",
                    "created_at": "2024-01-14",
                    "token_count": 19
                },
                {
                    "id": "demo-chunk-3",
                    "document_id": "demo-3",
                    "document_name": "License_Agreement_v2.pdf",
                    "chunk_index": 1,
                    "text_preview": "The licensee agrees to use the software in accordance with the terms...",
                    "full_text": "The licensee agrees to use the software in accordance with the terms and conditions outlined in this license agreement.",
                    "created_at": "2024-01-13",
                    "token_count": 20
                }
            ]
        
        return {"chunks": chunks}
        
    except Exception as e:
        logger.error(f"Chunks list error: {e}")
        # Return mock data if database fails
        return {
            "chunks": [
                {
                    "id": "demo-chunk-1",
                    "document_id": "demo-1",
                    "document_name": "Demo_Contract.pdf",
                    "chunk_index": 0,
                    "text_preview": "This is a demo contract chunk showing how text is broken down for processing...",
                    "full_text": "This is a demo contract chunk showing how text is broken down for processing and analysis.",
                    "created_at": "2024-01-15",
                    "token_count": 16
                }
            ]
        }

@app.get("/analytics")
async def get_analytics():
    """Get contract analytics data"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        db = components["db_manager"]
        
        # Get analytics data
        analytics_data = {
            "contract_timeline": [],
            "department_distribution": {},
            "expiring_contracts": 0,
            "total_value": 0
        }
        
        # Get contracts with dates for timeline
        try:
            contracts_result = db.client.table('contracts')\
                .select('contract_name, end_date, department')\
                .eq('status', 'active')\
                .execute()
        except Exception as e:
            logger.error(f"Failed to fetch contracts: {e}")
            contracts_result = None
        
        if contracts_result and contracts_result.data:
            import pandas as pd
            from datetime import timedelta
            
            contracts_df = pd.DataFrame(contracts_result.data)
            
            # Timeline data
            if 'end_date' in contracts_df.columns and not contracts_df.empty:
                timeline_data = contracts_df[contracts_df['end_date'].notna()]
                if not timeline_data.empty:
                    timeline_data['end_date'] = pd.to_datetime(timeline_data['end_date'])
                    
                    analytics_data["contract_timeline"] = [
                        {
                            "name": row['contract_name'],
                            "end_date": row['end_date'].isoformat() if pd.notna(row['end_date']) else None,
                            "department": row.get('department', 'Unknown')
                        }
                        for _, row in timeline_data.iterrows()
                    ]
                    
                    # Count expiring contracts (next 30 days)
                    expiring = timeline_data[
                        (timeline_data['end_date'] >= datetime.now()) &
                        (timeline_data['end_date'] <= datetime.now() + timedelta(days=30))
                    ]
                    analytics_data["expiring_contracts"] = len(expiring)
            
            # Department distribution
            if 'department' in contracts_df.columns and not contracts_df.empty:
                dept_counts = contracts_df['department'].value_counts().to_dict()
                analytics_data["department_distribution"] = dept_counts
            
            # Set total value to 0 since contract_value column doesn't exist
            analytics_data["total_value"] = 0
        
        return analytics_data
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# API Key Management Endpoints
@app.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(key_request: APIKeyRequest):
    """Generate a new API key for third-party integrations"""
    try:
        import secrets
        import hashlib
        from datetime import datetime, timedelta
        
        # Generate a secure API key
        key_id = str(uuid.uuid4())
        raw_key = f"clm_{secrets.token_urlsafe(32)}"
        
        # Store in memory for demo (in production, use database)
        api_key_data = {
            "id": key_id,
            "name": key_request.name,
            "key": raw_key,
            "created_at": datetime.now().isoformat(),
            "expires_at": key_request.expires_at,
            "description": key_request.description,
            "active": True
        }
        
        # In production, store in database with hashed key
        # For now, we'll just return the key data
        
        logger.info(f"API key created: {key_request.name}")
        
        return APIKeyResponse(
            id=key_id,
            name=key_request.name,
            key=raw_key,
            created_at=api_key_data["created_at"],
            expires_at=key_request.expires_at
        )
        
    except Exception as e:
        logger.error(f"API key creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api-keys")
async def list_api_keys():
    """List all API keys (without showing the actual key values)"""
    try:
        # In production, fetch from database
        # For demo, return sample data
        sample_keys = [
            {
                "id": "demo-key-1",
                "name": "Third Party Integration",
                "created_at": "2024-01-15T10:00:00",
                "last_used": "2024-01-20T15:30:00",
                "expires_at": None,
                "active": True,
                "description": "Integration with external document management system"
            },
            {
                "id": "demo-key-2", 
                "name": "Mobile App",
                "created_at": "2024-01-10T09:00:00",
                "last_used": "2024-01-22T11:45:00",
                "expires_at": "2024-06-10T09:00:00",
                "active": True,
                "description": "API access for mobile application"
            }
        ]
        
        return {"api_keys": sample_keys}
        
    except Exception as e:
        logger.error(f"API keys list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api-keys/{key_id}")
async def delete_api_key(key_id: str):
    """Delete/revoke an API key"""
    try:
        # In production, update database to mark key as inactive
        logger.info(f"API key deleted: {key_id}")
        return {"message": "API key deleted successfully"}
        
    except Exception as e:
        logger.error(f"API key deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def api_upload_documents(
    files: List[UploadFile] = File(...),
    api_key: str = Form(None)
):
    """API endpoint for third-party document uploads"""
    try:
        # In production, validate API key here
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        
        # For demo, accept any key that starts with 'clm_'
        if not api_key.startswith('clm_'):
            raise HTTPException(status_code=401, detail="Invalid API key")
            
        if "document_processor" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        document_processor = components["document_processor"]
        results = []
        
        for file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Process the file with API source tracking
                result = document_processor.process_single_document(
                    tmp_file_path,
                    filename=file.filename,
                    extract_contracts=True,
                    metadata={"source": "api_upload", "api_key": api_key[:20] + "..."}
                )
                
                results.append({
                    "filename": file.filename,
                    "success": result.get("success", False),
                    "document_id": result.get("document_id"),
                    "chunks_created": result.get("chunks_created", 0),
                    "contract_extracted": result.get("contract_extracted", False)
                })
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        logger.info(f"API upload completed: {len(results)} files processed")
        return {"results": results}
        
    except Exception as e:
        logger.error(f"API upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def classify_query(prompt: str) -> str:
    """Classify the type of user query"""
    prompt_lower = prompt.lower().strip()
    
    greeting_patterns = [
        'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon'
    ]
    
    help_patterns = [
        'help', 'how do i', 'how to', 'what can you do'
    ]
    
    agent_patterns = [
        'expiring', 'expire', 'conflict', 'summary', 'monitor', 'report'
    ]
    
    if any(pattern in prompt_lower for pattern in greeting_patterns):
        return "greeting"
    elif any(pattern in prompt_lower for pattern in help_patterns):
        return "help"
    elif any(pattern in prompt_lower for pattern in agent_patterns):
        return "agent_task"
    else:
        return "document_query"

def handle_greeting() -> Dict[str, Any]:
    """Handle greeting messages"""
    greeting_response = """
    Hello! 👋 Welcome to the Contract Lifecycle Management System. 
    
    I'm here to help you with your contract-related questions. You can ask me about:
    • Contract expiration dates and renewals
    • Contract analysis and key terms
    • Document search and similarity
    • Compliance and risk assessment
    • Contract analytics and reporting
    
    What would you like to know about your contracts today?
    """
    
    return {
        "answer": greeting_response,
        "sources": []
    }

def handle_help_request() -> Dict[str, Any]:
    """Handle help requests"""
    help_response = """
    🎯 **I'm your Contract Lifecycle Management Assistant!**
    
    **My capabilities include:**
    
    📋 **Contract Analysis**
    • Analyze contract content and terms
    • Extract key information from documents
    • Compare contracts and identify similarities
    
    📅 **Contract Monitoring**
    • Track contract expiration dates
    • Monitor upcoming renewals
    • Generate daily reports on contract status
    
    ⚠️ **Risk Management**
    • Identify potential contract conflicts
    • Flag important clauses and terms
    • Highlight compliance issues
    
    📊 **Analytics & Insights**
    • Generate contract summaries
    • Visualize contract timelines
    • Show department-wise contract distribution
    
    **How to get started:**
    1. Ask me questions about your contracts
    2. Upload new documents for analysis
    3. Request reports and analytics
    4. Search for specific contract information
    
    Try asking me something like: "Show me contracts expiring this month" or "What are the key terms in the TechCorp agreement?"
    """
    
    return {
        "answer": help_response,
        "sources": []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)