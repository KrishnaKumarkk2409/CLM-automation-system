"""
Streamlit chatbot interface for CLM automation system.
Provides interactive access to contract information via RAG pipeline.
"""

import streamlit as st
import logging
import sys
import os
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import json
import tempfile
from pathlib import Path
import re
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from src
# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv(override=True)

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
        logging.FileHandler('./logs/chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CLMChatbot:
    """Streamlit chatbot interface for contract management"""
    
    def __init__(self):
        """Initialize the chatbot with all necessary components"""
        self.initialize_components()
        self.greeting_responses = [
            "Hello! 👋 Welcome to the Contract Lifecycle Management System. I'm here to help you with your contract-related questions.",
            "Hi there! 😊 I'm your Contract Assistant. I can help you analyze contracts, find expiring agreements, and much more!",
            "Greetings! 🎯 Ready to dive into your contract management tasks? Ask me anything about your contracts!"
        ]
        
        self.sample_questions = [
            "📅 Show me contracts expiring in the next 30 days",
            "📊 What's the total value of all active contracts?",
            "⚠️ Are there any contract conflicts I should know about?",
            "🔍 Find all contracts with TechCorp",
            "📋 Give me a summary of our software licenses",
            "📈 Show contract analytics and department distribution",
            "🔄 What contracts need renewal soon?",
            "📝 What are the key terms in our NDA agreements?",
            "💰 Which contracts have the highest financial value?",
            "🏢 Show me all contracts by department"
        ]
    
    @st.cache_resource
    def initialize_components(_self):
        """Initialize database and AI components (cached)"""
        try:
            # Reload configuration to ensure fresh values
            Config.reload_config()
            
            # Check configuration first
            is_valid, missing = Config.validate_config()
            if not is_valid:
                st.error(f"❌ Configuration Error: Missing {', '.join(missing)}")
                st.info("Please check your .env file and ensure all required variables are set.")
                with st.expander("🔧 Configuration Help"):
                    st.markdown("""
                    **Required Environment Variables:**
                    - `SUPABASE_URL`: Your Supabase project URL
                    - `SUPABASE_KEY`: Your Supabase API key
                    - `OPENAI_API_KEY`: Your OpenAI API key
                    
                    **Optional Variables:**
                    - `EMAIL_USERNAME`, `EMAIL_PASSWORD`: For report notifications
                    - `DOCUMENTS_FOLDER`: Path to document storage (default: ./documents)
                    """)
                return None
            
            db_manager = DatabaseManager()
            embedding_manager = EmbeddingManager(db_manager)
            rag_pipeline = RAGPipeline(db_manager, embedding_manager)
            contract_agent = ContractAgent(db_manager)
            document_processor = DocumentProcessor(db_manager)
            
            return {
                "db_manager": db_manager,
                "embedding_manager": embedding_manager,
                "rag_pipeline": rag_pipeline,
                "contract_agent": contract_agent,
                "document_processor": document_processor
            }
        except Exception as e:
            st.error(f"❌ Failed to initialize CLM components: {str(e)}")
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Common Issues:**
                1. **Invalid OpenAI API Key**: Check if your API key is valid and has sufficient credits
                2. **Supabase Connection**: Verify your Supabase URL and key
                3. **Network Issues**: Ensure you have internet connectivity
                4. **Dependencies**: Run `pip install -r requirements.txt` to install missing packages
                """)
            return None
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Contract Lifecycle Management System",
            page_icon="📋",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for ChatGPT-like UI
        st.markdown("""
        <style>
        /* Hide default streamlit elements */
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        .stDecoration {display:none;}
        
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
            max-width: 900px;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Chat message styling - ChatGPT like */
        .user-message {
            background: #f7f7f8;
            padding: 1rem 1.5rem;
            margin: 0.5rem 0;
            border-radius: 18px;
            max-width: 70%;
            margin-left: auto;
            margin-right: 0;
            position: relative;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        .assistant-message {
            background: white;
            padding: 1rem 1.5rem;
            margin: 0.5rem 0;
            border-radius: 18px;
            max-width: 80%;
            margin-left: 0;
            margin-right: auto;
            border: 1px solid #e5e5e7;
            position: relative;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        /* Sample questions styling */
        .sample-question {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border: none;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: left;
            font-size: 0.9rem;
            color: #2d3748;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .sample-question:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            background: linear-gradient(135deg, #e2e8f0 0%, #a0aec0 100%);
        }
        
        /* Tools dropdown styling */
        .tools-dropdown {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 1000;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            padding: 0.5rem;
        }
        
        /* Chat container styling */
        .chat-container {
            height: 60vh;
            overflow-y: auto;
            padding: 1rem;
            border: 1px solid #e5e5e7;
            border-radius: 12px;
            background: #fafafa;
            margin-bottom: 1rem;
        }
        
        /* Input area styling */
        .stChatInput {
            border-radius: 25px !important;
            border: 2px solid #e5e5e7 !important;
            padding: 0.75rem 1rem !important;
        }
        
        .stChatInput:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Status indicators */
        .synthetic-docs-notice {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border: 2px solid #f6ad55;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            text-align: center;
            font-weight: 500;
            color: #744210;
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
            margin: 0.5rem;
        }
        
        .metric-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-1px);
            transition: all 0.3s ease;
        }
        
        /* Upload area */
        .upload-area {
            border: 2px dashed #cbd5e0;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
        }
        
        /* Hide streamlit menu and footer */
        #MainMenu {visibility: hidden;}
        .stSelectbox > div > div {
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "components" not in st.session_state:
            st.session_state.components = self.initialize_components()
            
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
            
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        if not st.session_state.components:
            st.error("❌ System initialization failed. Please check configuration.")
            if st.button("🔄 Retry Initialization"):
                st.session_state.pop('components', None)
                st.rerun()
            return
        
        # Main interface
        self.render_interface()
    
    def render_interface(self):
        """Render the ChatGPT-like Streamlit interface"""
        # Tools dropdown in top-right corner
        self.render_tools_dropdown()
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>📋 Contract Lifecycle Management System</h1>
            <p style="margin: 0; font-size: 1.1em; opacity: 0.9;">AI-powered contract analysis and management</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Synthetic documents notice
        self.render_synthetic_docs_notice()
        
        # System stats in a compact row
        self.render_compact_stats()
        
        # Show upload interface if requested
        if st.session_state.get('show_upload', False):
            self.render_document_upload()
            
            # Back to chat button
            if st.button("← Back to Chat", key="back_to_chat"):
                st.session_state.show_upload = False
                st.rerun()
        
        # Show analytics if requested
        elif st.session_state.get('show_analytics', False):
            self.render_insights_panel()
            
            # Back to chat button
            if st.button("← Back to Chat", key="back_to_chat_analytics"):
                st.session_state.show_analytics = False
                st.rerun()
        
        # Main chat interface (ChatGPT-like)
        else:
            self.render_chatgpt_interface()
    
    def render_tools_dropdown(self):
        """Render tools dropdown in top-right corner"""
        with st.sidebar:
            st.markdown("""
            <div style="position: fixed; top: 1rem; right: 1rem; z-index: 1000;">
                <details style="background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); padding: 0.5rem;">
                    <summary style="cursor: pointer; padding: 0.5rem; font-weight: bold; color: #667eea;">🛠️ Tools</summary>
                    <div style="padding: 0.5rem; min-width: 200px;">
                        <p><strong>📊 Analytics</strong></p>
                        <p><strong>📤 Upload Documents</strong></p>
                        <p><strong>📧 Generate Report</strong></p>
                        <p><strong>🔍 Search Documents</strong></p>
                        <p><strong>⚙️ Settings</strong></p>
                    </div>
                </details>
            </div>
            """, unsafe_allow_html=True)
            
            # Tools functionality
            st.subheader("🛠️ Tools & Actions")
            
            # Analytics
            if st.button("📊 View Analytics Dashboard", key="analytics_tool"):
                st.session_state.show_analytics = True
                st.rerun()
            
            # Document Upload
            if st.button("📤 Upload Documents", key="upload_tool"):
                st.session_state.show_upload = True
                st.rerun()
            
            # Generate Report
            if st.button("📧 Generate & Send Report", key="report_tool"):
                self.generate_report_with_email()
            
            # Document Processing
            if st.button("🔄 Process Folder Documents", key="process_tool"):
                with st.spinner("Processing documents..."):
                    components = st.session_state.components
                    results = components["document_processor"].process_folder()
                    st.success(f"Processed {len(results['processed'])} documents")
                    if results['failed']:
                        st.warning(f"Failed to process: {', '.join(results['failed'])}")
            
            # Document Search
            st.subheader("🔍 Document Search")
            search_query = st.text_input("Search documents:", key="doc_search")
            if search_query and st.button("Search", key="search_btn"):
                components = st.session_state.components
                similar = components["rag_pipeline"].find_similar_contracts(search_query)
                st.session_state.search_results = similar
                st.success(f"Found {len(similar)} similar documents")
            
            # Display search results if available
            if st.session_state.get('search_results'):
                st.subheader("📄 Search Results")
                results = st.session_state.search_results
                
                for i, doc in enumerate(results[:3]):  # Show top 3 in sidebar
                    with st.expander(f"📄 {doc['filename']} (Score: {doc['similarity']:.3f})"):
                        st.write(f"**Type:** {doc['file_type']}")
                        st.write(f"**Similarity:** {doc['similarity']:.3f}")
                        st.write(f"**Preview:** {doc['relevant_excerpt'][:150]}...")
                        
                        if st.button(f"Ask about this document", key=f"ask_doc_{i}"):
                            question = f"Tell me about the key points in {doc['filename']}"
                            st.session_state.messages.append({"role": "user", "content": question})
                            # Trigger a rerun to process the question
                            st.rerun()
                
                if len(results) > 3:
                    st.info(f"+ {len(results) - 3} more results. Check main area for full list.")
                
                if st.button("❌ Clear Search Results"):
                    st.session_state.pop('search_results', None)
                    st.rerun()
    
    def render_synthetic_docs_notice(self):
        """Show notice about synthetic documents being processed"""
        st.markdown("""
        <div class="synthetic-docs-notice">
            <h3 style="margin: 0 0 0.5rem 0; color: #744210;">🤖 Currently Processing Synthetic Contract Documents</h3>
            <p style="margin: 0; font-size: 0.9rem;">The system is loaded with sample contract data for demonstration. You can upload your own documents using the Tools menu or the Upload button below.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📤 Upload Your Documents", key="upload_main", type="primary"):
                st.session_state.show_upload = True
                st.rerun()
    
    def render_compact_stats(self):
        """Render compact system statistics"""
        try:
            components = st.session_state.components
            if not components:
                return
                
            db = components["db_manager"]
            
            # Get basic stats
            docs_result = db.client.table('documents').select('id').execute()
            total_docs = len(docs_result.data)
            
            contracts_result = db.client.table('contracts').select('id').eq('status', 'active').execute()
            active_contracts = len(contracts_result.data)
            
            chunks_result = db.client.table('document_chunks').select('id').execute()
            total_chunks = len(chunks_result.data)
            
            # Compact stats display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #667eea;">📄</h3>
                    <p style="margin: 0.25rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{}</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #666;">Documents</p>
                </div>
                """.format(total_docs), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #48bb78;">📋</h3>
                    <p style="margin: 0.25rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{}</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #666;">Active Contracts</p>
                </div>
                """.format(active_contracts), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #ed8936;">🔍</h3>
                    <p style="margin: 0.25rem 0 0 0; font-size: 1.5rem; font-weight: bold;">{}</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #666;">Text Chunks</p>
                </div>
                """.format(total_chunks), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #9f7aea;">🤖</h3>
                    <p style="margin: 0.25rem 0 0 0; font-size: 1.5rem; font-weight: bold;">GPT-4</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #666;">AI Model</p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            logger.error(f"Error rendering compact stats: {e}")
    
    def render_chatgpt_interface(self):
        """Render ChatGPT-like chat interface with left sidebar"""
        # Create main layout with left sidebar for tools
        left_col, main_col = st.columns([1, 3])
        
        with left_col:
            self._render_chat_sidebar()
        
        with main_col:
            # Show document preview if requested
            if st.session_state.get('show_document_preview', False):
                self._render_document_preview()
                return
            
            # Show sample questions if no conversation started
            if not st.session_state.messages:
                st.markdown("## 💭 Try asking me something like...")
                
                # Sample questions in a nice grid
                cols = st.columns(2)
                for i, question in enumerate(self.sample_questions[:8]):  # Show first 8
                    with cols[i % 2]:
                        if st.button(question, key=f"sample_{i}", help="Click to ask this question"):
                            self.handle_sample_question(question)
                
                # Show more samples in expandable section
                with st.expander("💡 More sample questions"):
                    cols2 = st.columns(2)
                    for i, question in enumerate(self.sample_questions[8:], 8):
                        with cols2[i % 2]:
                            if st.button(question, key=f"sample_{i}", help="Click to ask this question"):
                                self.handle_sample_question(question)
        
        # Chat history container
        if st.session_state.messages:
            st.markdown("## 💬 Conversation")
            
            # Chat history management buttons
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("🗑️ Clear", key="clear_chat_main"):
                    st.session_state.messages = []
                    st.rerun()
            
            with col2:
                if st.button("📥 Export", key="export_chat_main"):
                    if st.session_state.messages:
                        chat_export = {
                            "timestamp": datetime.now().isoformat(),
                            "messages": st.session_state.messages
                        }
                        st.download_button(
                            "💾 Download Chat",
                            data=json.dumps(chat_export, indent=2),
                            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            key="download_chat"
                        )
            
            # Chat messages with ChatGPT-like styling
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Handle newlines in assistant messages
                    content_with_breaks = message["content"].replace('\n', '<br>')
                    st.markdown(f"""
                    <div class="assistant-message">
                        <strong>🤖 Assistant:</strong><br>
                        {content_with_breaks}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources if available
                    if message.get("sources"):
                        self.display_sources(message["sources"])
        
        # Show search results if available
        if st.session_state.get('search_results'):
            st.markdown("## 🔍 Search Results")
            results = st.session_state.search_results
            
            # Create tabs for better organization
            if len(results) > 0:
                # Show all results in main area
                cols = st.columns(2)
                for i, doc in enumerate(results):
                    with cols[i % 2]:
                        with st.container():
                            st.markdown(f"""
                            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; background: white;">
                                <h4 style="margin: 0 0 0.5rem 0; color: #2e86ab;">📄 {doc['filename']}</h4>
                                <p style="margin: 0; font-size: 0.9rem; color: #666;"><strong>Type:</strong> {doc['file_type'].upper()}</p>
                                <p style="margin: 0; font-size: 0.9rem; color: #666;"><strong>Similarity:</strong> {doc['similarity']:.1%}</p>
                                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;"><strong>Preview:</strong></p>
                                <p style="margin: 0; font-size: 0.8rem; color: #444; font-style: italic;">{doc['relevant_excerpt'][:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"💬 Ask about this", key=f"ask_main_{i}"):
                                    question = f"Tell me about the key information in {doc['filename']}"
                                    st.session_state.messages.append({"role": "user", "content": question})
                                    st.rerun()
                            
                            with col2:
                                if st.button(f"🔍 Find similar", key=f"similar_main_{i}"):
                                    question = f"Find documents similar to {doc['filename']}"
                                    st.session_state.messages.append({"role": "user", "content": question})
                                    st.rerun()
                
                # Clear results button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("❌ Clear Search Results", key="clear_main_search"):
                        st.session_state.pop('search_results', None)
                        st.rerun()
        
        # Chat input at the bottom (ChatGPT style)
        st.markdown("---")
        
        # Input area
        prompt = st.chat_input("💬 Ask me anything about your contracts...")
        
        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response with dynamic spinner
            spinner_message = self._get_spinner_message(prompt)
            
            with st.spinner(spinner_message):
                response_data = self.get_ai_response(prompt)
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data["answer"],
                    "sources": response_data.get("sources", []),
                    "response_type": response_data.get("response_type", "unknown")
                })
            
            # Rerun to show new messages
            st.rerun()
    
    def _render_chat_sidebar(self):
        """Render left sidebar with tools and file upload"""
        st.markdown("### 🛠️ Tools")
        
        # File upload section
        st.markdown("#### 📤 Upload Files")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Drop files here", 
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'],
            help="Upload contracts, images, or documents",
            key="chat_uploader"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Ask if user wants to process files
            if st.button("📤 Process for Context", help="Process these files to add them to the knowledge base"):
                self._process_uploaded_files_for_context(uploaded_files)
            
            # Show uploaded file list
            with st.expander("View uploaded files"):
                for file in uploaded_files:
                    file_type = file.type.split('/')[-1] if file.type else 'unknown'
                    st.write(f"📄 {file.name} ({file_type})")
        
        st.divider()
        
        # Quick tools
        st.markdown("#### ⚡ Quick Tools")
        
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.show_analytics = True
            st.rerun()
        
        if st.button("🔍 Search Docs", use_container_width=True):
            search_query = st.text_input("Search:", key="quick_search")
            if search_query:
                components = st.session_state.components
                similar = components["rag_pipeline"].find_similar_contracts(search_query)
                st.session_state.search_results = similar
                st.success(f"Found {len(similar)} documents")
        
        if st.button("📧 Generate Report", use_container_width=True):
            self.generate_report_with_email()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            # Clear conversation memory
            if st.session_state.components:
                rag_pipeline = st.session_state.components["rag_pipeline"]
                rag_pipeline.clear_memory()
            st.session_state.messages = []
            st.rerun()
        
        # Show search results if available
        if st.session_state.get('search_results'):
            st.markdown("#### 🔍 Search Results")
            results = st.session_state.search_results[:3]  # Show top 3
            
            for i, doc in enumerate(results):
                with st.expander(f"📄 {doc['filename'][:20]}..."):
                    st.write(f"Score: {doc['similarity']:.1%}")
                    if st.button(f"Ask about this", key=f"sidebar_ask_{i}"):
                        question = f"Tell me about {doc['filename']}"
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
    
    def _process_uploaded_files_for_context(self, uploaded_files):
        """Process uploaded files and add to knowledge base"""
        if not st.session_state.components:
            st.error("System not initialized")
            return
        
        components = st.session_state.components
        document_processor = components["document_processor"]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed = 0
        failed = 0
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                # Check if it's an image file
                if uploaded_file.type and uploaded_file.type.startswith('image/'):
                    # Use OpenAI Vision for image analysis
                    result = self._analyze_image_with_vision(uploaded_file)
                else:
                    # Save uploaded file temporarily and process
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    result = document_processor.process_single_document(
                        tmp_file_path, 
                        filename=uploaded_file.name,
                        extract_contracts=True
                    )
                    
                    # Clean up
                    os.unlink(tmp_file_path)
                
                if result.get("success", False):
                    processed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {uploaded_file.name}: {e}")
                failed += 1
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        if processed > 0:
            st.success(f"✅ Successfully processed {processed} files")
        if failed > 0:
            st.warning(f"⚠️ Failed to process {failed} files")
        
        # Clear the progress indicators after a delay
        import time
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
    
    def _analyze_image_with_vision(self, image_file) -> Dict[str, Any]:
        """Analyze image using OpenAI Vision API"""
        try:
            import base64
            from openai import OpenAI
            
            # Convert image to base64
            image_data = base64.b64encode(image_file.getvalue()).decode('utf-8')
            
            client = OpenAI(api_key=Config.OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image for any contract-related information, text, signatures, or legal documents. Extract any readable text and describe what you see."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            analysis_text = response.choices[0].message.content
            
            # Store the analysis as a document
            components = st.session_state.components
            db_manager = components["db_manager"]
            
            document_id = db_manager.insert_document(
                filename=f"image_analysis_{image_file.name}",
                file_type="image_analysis",
                content=analysis_text,
                metadata={"original_filename": image_file.name, "analysis_type": "vision"}
            )
            
            return {
                "success": True,
                "document_id": document_id,
                "analysis": analysis_text
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _render_document_preview(self):
        """Render document preview modal"""
        if not st.session_state.get('document_preview'):
            return
        
        preview_data = st.session_state.document_preview
        
        st.markdown(f"## 📄 Document Preview: {preview_data['filename']}")
        
        # Close button
        if st.button("✕ Close Preview"):
            st.session_state.show_document_preview = False
            st.session_state.pop('document_preview', None)
            st.rerun()
        
        # Document metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Type", preview_data['file_type'].upper())
        with col2:
            st.metric("Content Length", f"{len(preview_data['content'])} chars")
        with col3:
            if preview_data.get('metadata', {}).get('total_pages'):
                st.metric("Pages", preview_data['metadata']['total_pages'])
        
        # Document content
        st.markdown("### 📜 Content")
        content = preview_data['content']
        
        # Show content in scrollable text area
        st.text_area(
            "Document Content",
            value=content,
            height=400,
            disabled=True,
            key="doc_preview_content"
        )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💬 Ask About This Document"):
                question = f"Analyze the document {preview_data['filename']} and provide key insights"
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.show_document_preview = False
                st.session_state.pop('document_preview', None)
                st.rerun()
        
        with col2:
            if st.button("🔍 Find Similar Documents"):
                # Use a sample of content to find similar docs
                sample_content = content[:500]
                components = st.session_state.components
                similar = components["rag_pipeline"].find_similar_contracts(sample_content)
                st.session_state.search_results = similar
                st.session_state.show_document_preview = False
                st.session_state.pop('document_preview', None)
                st.rerun()
        
        with col3:
            # Download button (placeholder)
            st.download_button(
                "📥 Download",
                data=content,
                file_name=preview_data['filename'],
                mime="text/plain"
            )
    
    def handle_sample_question(self, question):
        """Handle clicking on a sample question"""
        # Remove emoji from question for processing
        clean_question = question.split(' ', 1)[1] if ' ' in question else question
        
        # Add to messages
        st.session_state.messages.append({"role": "user", "content": clean_question})
        
        # Generate response
        response_data = self.get_ai_response(clean_question)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["answer"],
            "sources": response_data.get("sources", []),
            "response_type": response_data.get("response_type", "unknown")
        })
        
        st.rerun()
    
    def generate_report_with_email(self):
        """Generate report with email collection"""
        with st.form("email_report_form"):
            st.subheader("📧 Generate & Send Report")
            
            # Email input
            user_email = st.text_input(
                "📧 Your Email Address",
                value="",
                placeholder="Enter your email to receive the report",
                help="We'll send the contract report to this email address"
            )
            
            # Report options
            col1, col2 = st.columns(2)
            with col1:
                include_expiring = st.checkbox("Include expiring contracts", value=True)
                include_conflicts = st.checkbox("Include conflict analysis", value=True)
            
            with col2:
                include_analytics = st.checkbox("Include analytics", value=True)
                include_summaries = st.checkbox("Include contract summaries", value=False)
            
            # Submit button
            if st.form_submit_button("📤 Generate & Send Report", type="primary"):
                if not user_email:
                    st.error("Please enter your email address")
                elif not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', user_email):
                    st.error("Please enter a valid email address")
                else:
                    with st.spinner("Generating and sending report..."):
                        try:
                            components = st.session_state.components
                            contract_agent = components["contract_agent"]
                            
                            # Update the email in config temporarily
                            original_email = Config.REPORT_EMAIL
                            Config.REPORT_EMAIL = user_email
                            
                            # Generate report
                            report = contract_agent.generate_daily_report()
                            
                            # Send report (this will use the updated email)
                            contract_agent.send_report_email(report)
                            
                            # Restore original email
                            Config.REPORT_EMAIL = original_email
                            
                            st.success(f"✅ Report generated and sent to {user_email}!")
                        except Exception as e:
                            st.error(f"❌ Failed to send report: {str(e)}")
                            logger.error(f"Report generation failed: {e}")
    
    def render_sidebar(self):
        """Render the ChatGPT-like Streamlit interface"""
        # Tools dropdown in top-right corner
        self.render_tools_dropdown()
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>📋 Contract Lifecycle Management System</h1>
            <p style="margin: 0; font-size: 1.1em; opacity: 0.9;">AI-powered contract analysis and management</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Synthetic documents notice
        self.render_synthetic_docs_notice()
        
        # System stats in a compact row
        self.render_compact_stats()
        
        # Show upload interface if requested
        if st.session_state.get('show_upload', False):
            self.render_document_upload()
            
            # Back to chat button
            if st.button("← Back to Chat", key="back_to_chat"):
                st.session_state.show_upload = False
                st.rerun()
        
        # Show analytics if requested
        elif st.session_state.get('show_analytics', False):
            self.render_insights_panel()
            
            # Back to chat button
            if st.button("← Back to Chat", key="back_to_chat_analytics"):
                st.session_state.show_analytics = False
                st.rerun()
        
        # Main chat interface (ChatGPT-like)
        else:
            self.render_chatgpt_interface()
    
    def render_sidebar(self):
        """Render sidebar with system controls and information"""
        st.sidebar.header("🔧 System Controls")
        
        components = st.session_state.components
        
        # Document ingestion
        st.sidebar.subheader("📄 Document Management")
        
        if st.sidebar.button("Process Documents"):
            with st.sidebar:
                with st.spinner("Processing documents..."):
                    results = components["document_processor"].process_folder()
                    st.success(f"Processed {len(results['processed'])} documents")
                    if results['failed']:
                        st.warning(f"Failed to process: {', '.join(results['failed'])}")
        
        # Daily report generation
        st.sidebar.subheader("📊 Reports & Monitoring")
        
        if st.sidebar.button("Generate Daily Report"):
            with st.sidebar:
                with st.spinner("Generating report..."):
                    report = components["contract_agent"].generate_daily_report()
                    st.session_state.daily_report = report
                    st.success("Report generated!")
        
        if st.sidebar.button("Run Contract Monitoring"):
            with st.sidebar:
                with st.spinner("Running monitoring..."):
                    results = components["contract_agent"].run_daily_monitoring()
                    st.success(f"Monitoring complete: {results.get('status', 'unknown')}")
        
        # Document similarity
        st.sidebar.subheader("🔍 Document Similarity")
        
        similarity_query = st.sidebar.text_input("Find similar documents:")
        if similarity_query and st.sidebar.button("Search Similar"):
            with st.sidebar:
                with st.spinner("Searching..."):
                    similar = components["rag_pipeline"].find_similar_contracts(similarity_query)
                    st.session_state.similar_docs = similar
                    st.success(f"Found {len(similar)} similar documents")
        
        # System stats
        st.sidebar.subheader("📈 System Statistics")
        self.render_system_stats()
    
    def render_system_stats(self):
        """Render system statistics in sidebar"""
        try:
            components = st.session_state.components
            db = components["db_manager"]
            
            # Get basic stats
            docs_result = db.client.table('documents').select('id').execute()
            total_docs = len(docs_result.data)
            
            contracts_result = db.client.table('contracts').select('id').eq('status', 'active').execute()
            active_contracts = len(contracts_result.data)
            
            chunks_result = db.client.table('document_chunks').select('id').execute()
            total_chunks = len(chunks_result.data)
            
            # Display stats
            st.sidebar.metric("Total Documents", total_docs)
            st.sidebar.metric("Active Contracts", active_contracts)
            st.sidebar.metric("Text Chunks", total_chunks)
            
        except Exception as e:
            st.sidebar.error(f"Stats error: {e}")
    
    
    def _get_spinner_message(self, prompt: str) -> str:
        """Get appropriate spinner message based on query type"""
        query_type = self._classify_query(prompt)
        
        spinner_messages = {
            "greeting": "🤝 Preparing a warm welcome...",
            "help": "📚 Gathering information about my capabilities...",
            "agent_task": "🔍 Running contract analysis...",
            "document_query": "📄 Searching through contract documents..."
        }
        
        return spinner_messages.get(query_type, "🤔 Processing your request...")
    
    def render_insights_panel(self):
        """Render insights and visualization panel"""
        st.header("📊 Contract Insights")
        
        # Show daily report if generated
        if hasattr(st.session_state, 'daily_report'):
            st.subheader("📄 Latest Daily Report")
            report = st.session_state.daily_report
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expiring Contracts", report.get("expiring_contracts", 0))
            with col2:
                st.metric("Conflicts Found", report.get("conflicts_found", 0))
            
            # Report content (expandable)
            with st.expander("View Full Report", expanded=False):
                st.text(report.get("report_content", "No content available"))
        
        # Show similar documents if searched
        if hasattr(st.session_state, 'similar_docs'):
            st.subheader("🔍 Similar Documents")
            similar_docs = st.session_state.similar_docs
            
            for doc in similar_docs[:3]:  # Show top 3
                with st.expander(f"📄 {doc['filename']} (Similarity: {doc['similarity']:.3f})"):
                    st.write(f"**Type:** {doc['file_type']}")
                    st.write(f"**Excerpt:** {doc['relevant_excerpt']}")
        
        # Contract timeline visualization
        self.render_contract_timeline()
        
        # Department distribution
        self.render_department_chart()
    
    def render_contract_timeline(self):
        """Render contract expiration timeline"""
        try:
            st.subheader("📅 Contract Timeline")
            
            components = st.session_state.components
            db = components["db_manager"]
            
            # Get contracts with dates
            result = db.client.table('contracts')\
                .select('contract_name, end_date, department')\
                .not_.is_('end_date', 'null')\
                .eq('status', 'active')\
                .execute()
            
            if result.data:
                contracts_df = pd.DataFrame(result.data)
                contracts_df['end_date'] = pd.to_datetime(contracts_df['end_date'])
                contracts_df = contracts_df.sort_values('end_date')
                
                # Create timeline chart
                fig = px.scatter(contracts_df, 
                               x='end_date', 
                               y='contract_name',
                               color='department',
                               title='Contract Expiration Timeline',
                               hover_data=['department'])
                
                fig.add_vline(x=datetime.now(), line_dash="dash", line_color="red", 
                            annotation_text="Today")
                fig.add_vline(x=datetime.now() + timedelta(days=30), 
                            line_dash="dash", line_color="orange", 
                            annotation_text="30 days")
                
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Timeline error: {e}")
    
    def render_department_chart(self):
        """Render department distribution chart"""
        try:
            st.subheader("🏢 Contracts by Department")
            
            components = st.session_state.components
            db = components["db_manager"]
            
            # Get department distribution
            result = db.client.table('contracts')\
                .select('department')\
                .eq('status', 'active')\
                .execute()
            
            if result.data:
                dept_data = pd.DataFrame(result.data)
                dept_counts = dept_data['department'].value_counts()
                
                fig = px.pie(values=dept_counts.values, 
                           names=dept_counts.index,
                           title='Active Contracts by Department')
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Chart error: {e}")
    
    def get_ai_response(self, prompt: str) -> Dict[str, Any]:
        """Get AI response for user prompt"""
        try:
            components = st.session_state.components
            
            # Classify the type of query
            query_type = self._classify_query(prompt)
            
            if query_type == "greeting":
                return self._handle_greeting(prompt)
            elif query_type == "help":
                return self._handle_help_request(prompt)
            elif query_type == "agent_task":
                # Use contract agent for monitoring tasks
                response = components["contract_agent"].query_agent(prompt)
                return {
                    "answer": response,
                    "sources": [],
                    "response_type": "agent"
                }
            else:
                # Use RAG pipeline for document queries
                result = components["rag_pipeline"].query(prompt)
                return result
                
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return {
                "answer": f"I encountered an error processing your request: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _classify_query(self, prompt: str) -> str:
        """Classify the type of user query"""
        prompt_lower = prompt.lower().strip()
        
        # Greeting patterns
        greeting_patterns = [
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'howdy', 'what\'s up', 'whats up', 'sup'
        ]
        
        # Help patterns
        help_patterns = [
            'help', 'how do i', 'how to', 'what can you do', 'what are your capabilities',
            'how does this work', 'guide me', 'show me how', 'instructions'
        ]
        
        # Agent task patterns (contract monitoring)
        agent_patterns = [
            'expiring', 'expire', 'conflict', 'summary', 'monitor', 'report',
            'daily report', 'upcoming deadlines', 'contract status'
        ]
        
        # Check for greeting
        if any(pattern in prompt_lower for pattern in greeting_patterns):
            return "greeting"
        
        # Check for help request
        if any(pattern in prompt_lower for pattern in help_patterns):
            return "help"
            
        # Check for agent tasks
        if any(pattern in prompt_lower for pattern in agent_patterns):
            return "agent_task"
            
        # Default to document query
        return "document_query"
    
    def _handle_greeting(self, prompt: str) -> Dict[str, Any]:
        """Handle greeting messages with friendly responses"""
        import random
        
        greeting_response = random.choice(self.greeting_responses)
        
        # Add conversation starters
        starters_text = "\n\n**Here are some things I can help you with:**\n"
        for starter in self.sample_questions[:5]:  # Use sample_questions instead
            starters_text += f"• {starter}\n"
            
        full_response = greeting_response + starters_text
        
        return {
            "answer": full_response,
            "sources": [],
            "response_type": "greeting"
        }
    
    def _handle_help_request(self, prompt: str) -> Dict[str, Any]:
        """Handle help and capability requests"""
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
        
        📤 **Document Management**
        • Process new contract documents
        • Support PDF, DOCX, and TXT formats
        • Organize and index contract content
        
        **How to get started:**
        1. Ask me questions about your contracts
        2. Use the Quick Actions buttons for common tasks
        3. Upload new documents using the Upload tab
        4. Check the Insights tab for visualizations
        
        Try asking me something like: "Show me contracts expiring this month" or "What are the key terms in the TechCorp agreement?"
        """
        
        return {
            "answer": help_response,
            "sources": [],
            "response_type": "help"
        }
    
    
    def display_sources(self, sources: List[Dict[str, Any]]):
        """Display source citations with preview and links"""
        if not sources:
            return
        
        st.markdown("**📚 Sources:**")
        
        for i, source in enumerate(sources):
            with st.expander(f"📄 {source['filename']} (Similarity: {source.get('similarity', 'N/A')})", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Document ID:** {source.get('document_id', 'Unknown')}")
                    st.write(f"**Similarity Score:** {source.get('similarity', 0):.1%}")
                    st.write(f"**Relevant excerpt:**")
                    st.text_area("Content", value=source.get('chunk_text', 'No preview available'), height=100, key=f"source_text_{i}")
                
                with col2:
                    # Document actions
                    if st.button(f"🔍 View Full Doc", key=f"view_doc_{i}"):
                        self._show_document_preview(source.get('document_id'), source['filename'])
                    
                    if st.button(f"💬 Ask About This", key=f"ask_source_{i}"):
                        question = f"Tell me more about the information in {source['filename']}"
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
                    
                    if st.button(f"📥 Download", key=f"download_{i}"):
                        # Create download link (placeholder - you'd implement actual file serving)
                        st.info("Download functionality coming soon")
    
    def _show_document_preview(self, document_id: str, filename: str):
        """Show document preview in a modal or expander"""
        try:
            components = st.session_state.components
            if not components:
                st.error("System not initialized")
                return
            
            db = components["db_manager"]
            document = db.get_document_by_id(document_id)
            
            if document:
                st.session_state.document_preview = {
                    'filename': filename,
                    'content': document.get('content', ''),
                    'metadata': document.get('metadata', {}),
                    'file_type': document.get('file_type', 'unknown')
                }
                st.session_state.show_document_preview = True
                st.rerun()
            else:
                st.error("Document not found")
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
    
    def render_document_upload(self):
        """Render enhanced document upload interface with drag-and-drop functionality"""
        st.markdown("""
        <div class="upload-area">
            <h2 style="text-align: center; color: #2e86ab; margin-bottom: 0.5rem;">📤 Upload Contract Documents</h2>
            <p style="text-align: center; color: #666; margin: 0;">Drag and drop your files here, or click to browse</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader with enhanced UI
        uploaded_files = st.file_uploader(
            "Choose contract files", 
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT. Maximum file size: 200MB per file.",
            label_visibility="collapsed"
        )
        
        # Upload settings
        with st.expander("⚙️ Upload Settings"):
            col1, col2 = st.columns(2)
            with col1:
                auto_process = st.checkbox("🔄 Auto-process after upload", value=True, 
                                         help="Automatically process documents after upload")
                extract_contracts = st.checkbox("📋 Extract contract details", value=True,
                                               help="Automatically extract contract information")
            with col2:
                chunk_size = st.slider("🗖️ Text chunk size", min_value=500, max_value=2000, value=1000,
                                     help="Size of text chunks for processing")
                notify_completion = st.checkbox("🔔 Notify on completion", value=False,
                                               help="Send notification when processing is complete")
        
        # Display uploaded files
        if uploaded_files:
            st.subheader(f"📎 Uploaded Files ({len(uploaded_files)})")
            
            total_size = 0
            file_details = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                file_size_mb = uploaded_file.size / (1024 * 1024)
                total_size += file_size_mb
                
                file_details.append({
                    "Name": uploaded_file.name,
                    "Size": f"{file_size_mb:.2f} MB",
                    "Type": uploaded_file.type,
                    "Status": "🔄 Ready to process"
                })
            
            # Display file table
            files_df = pd.DataFrame(file_details)
            st.dataframe(files_df, use_container_width=True, hide_index=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Files", len(uploaded_files))
            with col2:
                st.metric("📊 Total Size", f"{total_size:.2f} MB")
            with col3:
                avg_size = total_size / len(uploaded_files) if uploaded_files else 0
                st.metric("📅 Avg Size", f"{avg_size:.2f} MB")
            
            st.divider()
            
            # Processing controls
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                    self._process_uploaded_files(uploaded_files, auto_process, extract_contracts, chunk_size)
            
            with col2:
                if st.button("📋 Preview Files", use_container_width=True):
                    self._preview_uploaded_files(uploaded_files)
            
            with col3:
                if st.button("❌ Clear All", use_container_width=True):
                    st.session_state.uploaded_files = []
                    st.rerun()
        
        # Upload history
        self._render_upload_history()
        
        # Tips for better uploads
        with st.expander("💡 Tips for Better Document Processing"):
            st.markdown("""
            **For best results:**
            • **PDF files**: Ensure they contain selectable text (not scanned images)
            • **DOCX files**: Use standard formatting, avoid complex layouts
            • **TXT files**: Use UTF-8 encoding for special characters
            • **File naming**: Use descriptive names (e.g., "TechCorp_License_2024.pdf")
            • **File size**: Smaller files process faster, consider splitting large documents
            • **Language**: Currently optimized for English contracts
            
            **Supported content types:**
            • Service agreements and contracts
            • Software licenses
            • Non-disclosure agreements (NDAs)
            • Maintenance agreements
            • Amendment documents
            """)
    
    def _process_uploaded_files(self, uploaded_files, auto_process: bool, extract_contracts: bool, chunk_size: int):
        """Process uploaded files with progress tracking"""
        if not st.session_state.components:
            st.error("❌ System not initialized. Please refresh the page.")
            return
        
        components = st.session_state.components
        document_processor = components["document_processor"]
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        processed_files = []
        failed_files = []
        
        try:
            for i, uploaded_file in enumerate(uploaded_files):
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"📄 Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process the file
                    result = document_processor.process_single_document(
                        tmp_file_path, 
                        filename=uploaded_file.name,
                        extract_contracts=extract_contracts,
                        custom_chunk_size=chunk_size
                    )
                    
                    if result.get("success", False):
                        processed_files.append({
                            "name": uploaded_file.name,
                            "document_id": result.get("document_id"),
                            "chunks_created": result.get("chunks_created", 0),
                            "contract_extracted": result.get("contract_extracted", False)
                        })
                    else:
                        failed_files.append(uploaded_file.name)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    logger.error(f"Failed to process {uploaded_file.name}: {e}")
                    failed_files.append(uploaded_file.name)
            
            # Update progress
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # Show results
            with results_container.container():
                st.success(f"✅ Successfully processed {len(processed_files)} out of {len(uploaded_files)} files")
                
                if processed_files:
                    st.subheader("✅ Successfully Processed")
                    success_df = pd.DataFrame(processed_files)
                    st.dataframe(success_df, use_container_width=True, hide_index=True)
                
                if failed_files:
                    st.subheader("❌ Failed to Process")
                    for failed_file in failed_files:
                        st.error(f"❌ {failed_file}")
                
                # Add to upload history
                upload_record = {
                    "timestamp": datetime.now(),
                    "processed_files": len(processed_files),
                    "failed_files": len(failed_files),
                    "total_files": len(uploaded_files)
                }
                
                if "upload_history" not in st.session_state:
                    st.session_state.upload_history = []
                st.session_state.upload_history.append(upload_record)
                
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            logger.error(f"Document processing failed: {e}")
        
        finally:
            # Clear progress indicators after a delay
            import time
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()
    
    def _preview_uploaded_files(self, uploaded_files):
        """Preview uploaded files content"""
        st.subheader("🔍 File Preview")
        
        # File selector for preview
        if len(uploaded_files) > 1:
            selected_file_name = st.selectbox(
                "Select file to preview:",
                [f.name for f in uploaded_files]
            )
            selected_file = next(f for f in uploaded_files if f.name == selected_file_name)
        else:
            selected_file = uploaded_files[0]
        
        try:
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", selected_file.name)
            with col2:
                st.metric("File Size", f"{selected_file.size / 1024:.1f} KB")
            with col3:
                st.metric("File Type", selected_file.type)
            
            # Preview content based on file type
            if selected_file.type == "text/plain":
                content = selected_file.getvalue().decode('utf-8')
                st.text_area("File Content (first 2000 characters):", 
                           value=content[:2000] + ("..." if len(content) > 2000 else ""),
                           height=300)
            
            elif selected_file.type == "application/pdf":
                st.info("📄 PDF file detected. Full content will be extracted during processing.")
                # Could add PDF preview using PyPDF2 or similar
                
            elif selected_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                st.info("📄 Word document detected. Full content will be extracted during processing.")
                # Could add DOCX preview using python-docx
            
            else:
                st.warning(f"⚠️ Unknown file type: {selected_file.type}")
                
        except Exception as e:
            st.error(f"❌ Preview failed: {str(e)}")
    
    def _render_upload_history(self):
        """Render upload history section"""
        if "upload_history" in st.session_state and st.session_state.upload_history:
            st.subheader("📅 Recent Upload History")
            
            history_data = []
            for record in st.session_state.upload_history[-5:]:  # Show last 5 uploads
                history_data.append({
                    "Date": record["timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "Total Files": record["total_files"],
                    "Processed": record["processed_files"],
                    "Failed": record["failed_files"],
                    "Success Rate": f"{(record['processed_files'] / record['total_files'] * 100):.1f}%" if record["total_files"] > 0 else "0%"
                })
            
            if history_data:
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True, hide_index=True)
                
                if st.button("🗋 Clear History"):
                    st.session_state.upload_history = []
                    st.rerun()

def main():
    """Main function to run the Streamlit app"""
    chatbot = CLMChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()