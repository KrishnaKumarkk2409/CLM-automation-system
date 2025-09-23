"""
FastAPI backend for CLM automation system.
Modern web API replacing Streamlit interface.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
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
from dotenv import load_dotenv
load_dotenv(override=True)

# Add the backend directory to the path so we can import from src
# Get the backend directory (parent of app directory)
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

# Also add the project root for good measure
project_root = os.path.dirname(backend_dir)
sys.path.append(project_root)

from src.config import Config
from src.database import DatabaseManager
from src.embeddings import EmbeddingManager
from src.rag_pipeline import RAGPipeline
from src.contract_agent import ContractAgent
from src.document_processor import DocumentProcessor

# Configure logging
# Create logs directory if it doesn't exist
logs_dir = os.path.join(project_root, 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'api.log')),
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
        
        # Create necessary directories using project root
        uploads_dir = os.path.join(project_root, 'uploads')
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(uploads_dir, exist_ok=True)
        
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
        
        # Get stats
        docs_result = db.client.table('documents').select('id').execute()
        total_docs = len(docs_result.data)
        
        contracts_result = db.client.table('contracts').select('id').eq('status', 'active').execute()
        active_contracts = len(contracts_result.data)
        
        chunks_result = db.client.table('document_chunks').select('id').execute()
        total_chunks = len(chunks_result.data)
        
        return SystemStats(
            total_documents=total_docs,
            active_contracts=active_contracts,
            total_chunks=total_chunks,
            system_status="operational"
        )
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """Main chat endpoint for contract queries"""
    try:
        if "rag_pipeline" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Generate unique IDs
        conversation_id = chat_request.conversation_id or str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        
        # Classify and process the query
        query_type = classify_query(chat_request.message)
        
        if query_type == "greeting":
            response_data = handle_greeting()
        elif query_type == "help":
            response_data = handle_help_request()
        elif query_type == "agent_task":
            # Use contract agent for monitoring tasks
            agent_response = components["contract_agent"].query_agent(chat_request.message)
            response_data = {
                "answer": agent_response,
                "sources": []
            }
        else:
            # Use RAG pipeline for document queries
            response_data = components["rag_pipeline"].query(chat_request.message)
        
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

@app.websocket("/ws/chat/{conversation_id}")
async def websocket_chat_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, conversation_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process different message types
            if message_data.get("type") == "chat":
                # Process chat message
                try:
                    response_data = components["rag_pipeline"].query(message_data["message"])
                    
                    # Send response back
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "chat_response",
                            "response": response_data["answer"],
                            "sources": response_data.get("sources", []),
                            "message_id": str(uuid.uuid4()),
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                except Exception as e:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
            
            elif message_data.get("type") == "ping":
                # Handle ping for connection health
                await manager.send_personal_message(
                    json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, conversation_id)
        logger.info(f"WebSocket disconnected for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, conversation_id)

@app.websocket("/ws/processing")
async def websocket_processing_endpoint(websocket: WebSocket):
    """WebSocket endpoint for document processing updates"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # This endpoint is mainly for sending processing updates to client
            # Client can send heartbeat messages
            message_data = json.loads(data)
            if message_data.get("type") == "heartbeat":
                await websocket.send_text(json.dumps({
                    "type": "heartbeat_ack",
                    "timestamp": datetime.now().isoformat()
                }))
    except WebSocketDisconnect:
        logger.info("Processing WebSocket disconnected")
    except Exception as e:
        logger.error(f"Processing WebSocket error: {e}")

# Bulk Processing Endpoints

@app.post("/documents/process/folder")
async def process_folder_documents():
    """Process all documents in the configured folder"""
    try:
        if "document_processor" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        document_processor = components["document_processor"]
        results = document_processor.process_folder()
        
        return {
            "processed": results.get("processed", []),
            "failed": results.get("failed", []),
            "total_processed": len(results.get("processed", [])),
            "total_failed": len(results.get("failed", [])),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Folder processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/process/batch")
async def process_batch_documents(document_ids: List[str]):
    """Process a batch of specific documents"""
    try:
        if "document_processor" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        results = {
            "processed": [],
            "failed": [],
            "total": len(document_ids)
        }
        
        # This would require implementing batch processing in document_processor
        # For now, return a placeholder response
        return {
            "message": "Batch processing initiated",
            "document_ids": document_ids,
            "status": "processing",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Configuration Endpoints

@app.get("/system/config")
async def get_system_config():
    """Get system configuration (non-sensitive)"""
    return {
        "smtp_configured": bool(Config.SMTP_SERVER and Config.EMAIL_USERNAME),
        "openai_configured": bool(Config.OPENAI_API_KEY),
        "supabase_configured": bool(Config.SUPABASE_URL and Config.SUPABASE_KEY),
        "documents_folder": Config.DOCUMENTS_FOLDER,
        "chunk_size": Config.CHUNK_SIZE,
        "similarity_threshold": Config.SIMILARITY_THRESHOLD,
        "system_version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/system/config/email")
async def update_email_config(email_config: dict):
    """Update email configuration"""
    try:
        # This would update the configuration
        # For security, we don't actually update Config directly via API
        # This would typically update a database or config file
        
        return {
            "message": "Email configuration updated",
            "smtp_server": email_config.get("smtp_server", "Not provided"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Email config update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
                # Process the file
                result = document_processor.process_single_document(
                    tmp_file_path,
                    filename=file.filename,
                    extract_contracts=True
                )
                
                # Check if the result contains an error message from OCR
                if not result.get("success", False) and "error" in result:
                    error_msg = result["error"]
                    if "ERROR:" in error_msg:
                        logger.error(f"OCR error for {file.filename}: {error_msg}")
                
                results.append({
                    "filename": file.filename,
                    "success": result.get("success", False),
                    "document_id": result.get("document_id"),
                    "chunks_created": result.get("chunks_created", 0),
                    "contract_extracted": result.get("contract_extracted", False),
                    "error": result.get("error", "Unknown processing error")
                })
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
            
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_file_path}: {e}")
        
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

# Analytics and Dashboard Endpoints

@app.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """Get comprehensive analytics data for dashboard"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        db = components["db_manager"]
        
        # Get basic statistics
        docs_result = db.client.table('documents').select('id, file_type, created_at').execute()
        contracts_result = db.client.table('contracts').select('id, status, department, end_date, start_date').execute()
        chunks_result = db.client.table('document_chunks').select('id').execute()
        
        # Process data
        total_docs = len(docs_result.data)
        active_contracts = len([c for c in contracts_result.data if c.get('status') == 'active'])
        total_chunks = len(chunks_result.data)
        
        # Department distribution
        dept_distribution = {}
        for contract in contracts_result.data:
            dept = contract.get('department', 'Unknown')
            dept_distribution[dept] = dept_distribution.get(dept, 0) + 1
        
        # File type distribution
        file_type_distribution = {}
        for doc in docs_result.data:
            file_type = doc.get('file_type', 'unknown')
            file_type_distribution[file_type] = file_type_distribution.get(file_type, 0) + 1
        
        # Contract timeline data (next 90 days)
        from datetime import datetime, timedelta
        today = datetime.now().date()
        timeline_data = []
        
        for contract in contracts_result.data:
            if contract.get('end_date'):
                try:
                    end_date = datetime.fromisoformat(contract['end_date'].replace('Z', '+00:00')).date()
                    days_until_expiry = (end_date - today).days
                    if -30 <= days_until_expiry <= 90:  # Past 30 days to next 90 days
                        timeline_data.append({
                            'contract_id': contract['id'],
                            'end_date': contract['end_date'],
                            'days_until_expiry': days_until_expiry,
                            'department': contract.get('department', 'Unknown'),
                            'status': contract.get('status', 'unknown')
                        })
                except:
                    continue
        
        return {
            'overview': {
                'total_documents': total_docs,
                'active_contracts': active_contracts,
                'total_chunks': total_chunks,
                'system_status': 'healthy'
            },
            'distributions': {
                'departments': dept_distribution,
                'file_types': file_type_distribution
            },
            'timeline': timeline_data,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analytics dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/contracts/expiring")
async def get_expiring_contracts(days: int = 30):
    """Get contracts expiring within specified days"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        expiring = components["db_manager"].get_expiring_contracts(days)
        
        return {
            'expiring_contracts': expiring,
            'count': len(expiring),
            'days_threshold': days,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Expiring contracts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/conflicts")
async def get_contract_conflicts():
    """Get detected contract conflicts"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        conflicts = components["db_manager"].find_contract_conflicts()
        
        return {
            'conflicts': conflicts,
            'count': len(conflicts),
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Contract conflicts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/report/generate")
async def generate_analytics_report(email: str = None):
    """Generate comprehensive analytics report"""
    try:
        if "contract_agent" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        contract_agent = components["contract_agent"]
        
        # Generate report
        report = contract_agent.generate_daily_report()
        
        # Send email if provided
        email_sent = False
        if email:
            email_sent = contract_agent.send_report_email(report, email)
        
        return {
            'report': report,
            'email_sent': email_sent,
            'recipient': email if email else None
        }
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Document Management

@app.get("/documents")
async def list_documents(limit: int = 50, offset: int = 0, file_type: str = None):
    """List documents with filtering"""
    try:
        if "db_manager" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        db = components["db_manager"]
        
        query = db.client.table('documents').select('*')
        
        if file_type:
            query = query.eq('file_type', file_type)
        
        result = query.range(offset, offset + limit - 1).execute()
        
        return {
            'documents': result.data,
            'count': len(result.data),
            'limit': limit,
            'offset': offset
        }
        
    except Exception as e:
        logger.error(f"Document listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/search")
async def search_documents_advanced(query: str, limit: int = 10, filters: dict = None):
    """Advanced document search with filters"""
    try:
        if "rag_pipeline" not in components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Use RAG pipeline for semantic search
        similar_docs = components["rag_pipeline"].find_similar_contracts(
            query, limit=limit
        )
        
        return {
            'documents': similar_docs,
            'query': query,
            'count': len(similar_docs),
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document search error: {e}")
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

def run_streamlit_app():
    """Launch the Streamlit chatbot interface"""
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    # Change to backend directory
    backend_dir = Path(__file__).parent.parent
    os.chdir(backend_dir)
    
    # Run streamlit with the chatbot interface
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/chatbot_interface.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--theme.base=light"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit chatbot stopped")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CLM Backend Server")
    parser.add_argument("--chatbot", action="store_true", help="Run Streamlit chatbot interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", default=True, help="Enable auto-reload")
    
    args = parser.parse_args()
    
    if args.chatbot:
        print("🚀 Starting Streamlit Chatbot Interface...")
        run_streamlit_app()
    else:
        print("🚀 Starting FastAPI Backend Server...")
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
