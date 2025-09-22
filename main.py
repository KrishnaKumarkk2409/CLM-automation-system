"""
Main application runner for CLM automation system.
Provides CLI interface and orchestrates all system components.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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
        logging.FileHandler('./logs/clm_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CLMSystem:
    """Main CLM automation system orchestrator"""
    
    def __init__(self):
        """Initialize all system components"""
        logger.info("Initializing CLM automation system...")
        
        # Validate configuration
        is_valid, missing = Config.validate_config()
        if not is_valid:
            logger.error(f"Configuration validation failed: Missing {missing}")
            print(f"❌ Configuration Error: Missing required settings: {', '.join(missing)}")
            print("   Please check your .env file and ensure all required variables are set.")
            raise ValueError(f"Missing configuration: {missing}")
        
        try:
            self.db_manager = DatabaseManager()
            self.embedding_manager = EmbeddingManager(self.db_manager)
            self.rag_pipeline = RAGPipeline(self.db_manager, self.embedding_manager)
            self.contract_agent = ContractAgent(self.db_manager)
            self.document_processor = DocumentProcessor(self.db_manager)
            
            logger.info("CLM system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLM system: {e}")
            raise
    
    def process_documents(self, folder_path: str = None) -> Dict[str, Any]:
        """Process documents in the specified folder"""
        folder_path = folder_path or Config.DOCUMENTS_FOLDER
        logger.info(f"Processing documents in folder: {folder_path}")
        
        try:
            results = self.document_processor.process_folder(folder_path)
            
            print(f"\n📄 Document Processing Results:")
            print(f"✅ Successfully processed: {len(results['processed'])} documents")
            print(f"❌ Failed to process: {len(results['failed'])} documents") 
            print(f"📊 Total chunks created: {results['total_chunks']}")
            
            if results['processed']:
                print("\n📋 Processed Documents:")
                for doc in results['processed']:
                    print(f"  • {doc['filename']} ({doc['chunks']} chunks)")
            
            if results['failed']:
                print("\n⚠️ Failed Documents:")
                for filename in results['failed']:
                    print(f"  • {filename}")
            
            return results
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            print(f"❌ Document processing failed: {e}")
            return {"error": str(e)}
    
    def query_contracts(self, question: str) -> Dict[str, Any]:
        """Query contracts using RAG pipeline"""
        logger.info(f"Processing contract query: {question[:100]}...")
        
        try:
            result = self.rag_pipeline.query(question)
            
            print(f"\n💬 Query: {question}")
            print(f"\n🤖 Answer:\n{result['answer']}")
            
            if result.get('sources'):
                print(f"\n📚 Sources ({len(result['sources'])} documents):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['filename']} (Similarity: {source['similarity']:.3f})")
                    print(f"     Preview: {source['chunk_text'][:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Contract query failed: {e}")
            print(f"❌ Query failed: {e}")
            return {"error": str(e)}
    
    def find_similar_documents(self, document_name: str = None, text_query: str = None) -> List[Dict[str, Any]]:
        """Find similar documents based on document name or text query"""
        logger.info(f"Finding similar documents for: {document_name or text_query}")
        
        try:
            if document_name:
                # Find document by name
                result = self.db_manager.client.table('documents')\
                    .select('*')\
                    .ilike('filename', f'%{document_name}%')\
                    .execute()
                
                if not result.data:
                    print(f"❌ Document not found: {document_name}")
                    return []
                
                document = result.data[0]
                similar_docs = self.db_manager.get_similar_documents(document['id'])
                
                print(f"\n🔍 Similar documents to '{document['filename']}':")
                
            elif text_query:
                # Find similar documents using text query
                similar_docs = self.rag_pipeline.find_similar_contracts(text_query)
                print(f"\n🔍 Documents similar to: '{text_query}':")
            
            else:
                print("❌ Please provide either a document name or text query")
                return []
            
            if not similar_docs:
                print("No similar documents found.")
                return []
            
            # Display results
            for i, doc in enumerate(similar_docs, 1):
                print(f"\n  {i}. 📄 {doc['filename']}")
                print(f"     Similarity: {doc.get('similarity', 'N/A'):.3f}")
                print(f"     Type: {doc.get('file_type', 'Unknown')}")
                if 'relevant_excerpt' in doc:
                    print(f"     Excerpt: {doc['relevant_excerpt'][:100]}...")
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Similar document search failed: {e}")
            print(f"❌ Similar document search failed: {e}")
            return []
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily contract monitoring report"""
        logger.info("Generating daily contract report...")
        
        try:
            report = self.contract_agent.generate_daily_report()
            
            print(f"\n📊 Daily Contract Report - {datetime.now().strftime('%Y-%m-%d')}")
            print("=" * 60)
            print(report['report_content'])
            
            print(f"\n📈 Summary:")
            print(f"  • Expiring contracts: {report.get('expiring_contracts', 0)}")
            print(f"  • Conflicts found: {report.get('conflicts_found', 0)}")
            
            return report
            
        except Exception as e:
            logger.error(f"Daily report generation failed: {e}")
            print(f"❌ Daily report generation failed: {e}")
            return {"error": str(e)}
    
    def run_monitoring(self, send_email: bool = False) -> Dict[str, Any]:
        """Run complete daily monitoring cycle"""
        logger.info("Running daily contract monitoring...")
        
        try:
            results = self.contract_agent.run_daily_monitoring()
            
            print(f"\n🔍 Daily Monitoring Results:")
            print(f"Status: {results['status'].upper()}")
            print(f"Report Generated: {'✅' if results['report_generated'] else '❌'}")
            print(f"Email Sent: {'✅' if results['email_sent'] else '❌'}")
            print(f"Expiring Contracts: {results.get('expiring_contracts', 0)}")
            print(f"Conflicts Found: {results.get('conflicts_found', 0)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Daily monitoring failed: {e}")
            print(f"❌ Daily monitoring failed: {e}")
            return {"error": str(e)}
    
    def interactive_mode(self):
        """Run interactive CLI mode"""
        print("🤖 CLM Interactive Mode")
        print("Type 'help' for available commands, 'quit' to exit")
        
        while True:
            try:
                user_input = input("\nCLM> ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye! 👋")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                
                elif user_input.lower() == 'process':
                    self.process_documents()
                
                elif user_input.lower() == 'report':
                    self.generate_daily_report()
                
                elif user_input.lower() == 'monitor':
                    self.run_monitoring()
                
                elif user_input.lower() == 'reload':
                    print("⚙️  Reloading configuration...")
                    if Config.reload_config():
                        print("✅ Configuration reloaded successfully")
                    else:
                        print("❌ Configuration reload failed")
                
                elif user_input.startswith('query '):
                    question = user_input[6:]  # Remove 'query '
                    self.query_contracts(question)
                
                elif user_input.startswith('similar '):
                    query = user_input[8:]  # Remove 'similar '
                    self.find_similar_documents(text_query=query)
                
                elif user_input.startswith('find '):
                    doc_name = user_input[5:]  # Remove 'find '
                    self.find_similar_documents(document_name=doc_name)
                
                else:
                    # Treat as natural language query
                    if user_input:
                        self.query_contracts(user_input)
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show help message"""
        print("""
🔧 Available Commands:
  help              - Show this help message
  process           - Process documents in the documents folder
  report            - Generate daily contract report
  monitor           - Run daily monitoring with alerts
  reload            - Reload configuration from .env file
  query <question>  - Ask a question about contracts
  similar <text>    - Find documents similar to given text
  find <filename>   - Find documents similar to a specific file
  quit/exit         - Exit the application
  
💡 You can also ask natural language questions directly!
   Example: "What contracts are expiring soon?"
        """)

def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(description='CLM Automation System')
    parser.add_argument('--process', action='store_true', help='Process documents')
    parser.add_argument('--folder', type=str, help='Folder path for document processing')
    parser.add_argument('--query', type=str, help='Query contracts')
    parser.add_argument('--report', action='store_true', help='Generate daily report')
    parser.add_argument('--monitor', action='store_true', help='Run daily monitoring')
    parser.add_argument('--similar', type=str, help='Find similar documents to text')
    parser.add_argument('--find', type=str, help='Find similar documents to file')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--chatbot', action='store_true', help='Start Streamlit chatbot')
    parser.add_argument('--reload-config', action='store_true', help='Reload configuration from .env file')
    
    args = parser.parse_args()
    
    try:
        # Create necessary directories
        os.makedirs('./logs', exist_ok=True)
        
        # Initialize system
        clm_system = CLMSystem()
        
        if args.reload_config:
            print("⚙️  Reloading configuration from .env file...")
            if Config.reload_config():
                print("✅ Configuration reloaded successfully")
                is_valid, _ = Config.validate_config()
                if is_valid:
                    print("✅ All required configuration present")
                else:
                    print("⚠️  Some configuration still missing")
            else:
                print("❌ Configuration reload failed")
        
        elif args.chatbot:
            print("🚀 Starting Streamlit chatbot interface...")
            os.system("python run_chatbot.py")
        
        elif args.process:
            clm_system.process_documents(args.folder)
        
        elif args.query:
            clm_system.query_contracts(args.query)
        
        elif args.report:
            clm_system.generate_daily_report()
        
        elif args.monitor:
            clm_system.run_monitoring()
        
        elif args.similar:
            clm_system.find_similar_documents(text_query=args.similar)
        
        elif args.find:
            clm_system.find_similar_documents(document_name=args.find)
        
        elif args.interactive:
            clm_system.interactive_mode()
        
        else:
            # Default to interactive mode
            print("🎉 Welcome to CLM Automation System!")
            clm_system.interactive_mode()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()