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
                # Process the file
                result = document_processor.process_single_document(
                    tmp_file_path,
                    filename=file.filename,
                    extract_contracts=True
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)