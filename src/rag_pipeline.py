"""
RAG (Retrieval-Augmented Generation) pipeline for CLM automation system.
Implements document retrieval and question answering using LangChain.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from collections import deque

from src.config import Config
from src.database import DatabaseManager
from src.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG pipeline for contract document retrieval and question answering"""
    
    def __init__(self, db_manager: DatabaseManager, embedding_manager: EmbeddingManager):
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        
        # Initialize simple conversation memory
        self.chat_history = deque(maxlen=10)  # Remember last 10 messages
        
        # Initialize OpenAI chat model
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=Config.OPENAI_API_KEY
        )
        
        # Define prompt template for contract queries with memory
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
You are a legal assistant specializing in contract analysis. You have access to conversation history and can reference previous discussions.

Analyze the user's question first to understand:
1. What specific information they need
2. How it relates to previous conversation
3. What context would be most helpful

Then provide a comprehensive answer based on the provided context.

Context from relevant contract documents:
{context}

Instructions:
- Answer based on the provided context and conversation history
- If the context doesn't contain enough information, say "I don't have sufficient information to answer this question based on the available contract documents."
- Always cite source documents when providing information
- Be precise and professional in your response
- For dates, amounts, or specific terms, quote directly from the documents
- Reference previous conversation when relevant
- Provide actionable insights when possible
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        logger.info("RAG pipeline initialized with GPT-4")
    
    def query(self, question: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Process a query using RAG pipeline
        
        Args:
            question: User's question about contracts
            max_results: Maximum number of document chunks to retrieve
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        try:
            logger.info(f"Processing RAG query: {question[:100]}...")
            
            # Step 1: Generate query embedding
            query_embedding = self.embedding_manager.generate_query_embedding(question)
            if not query_embedding:
                return {
                    "answer": "I'm sorry, I couldn't process your question at this time.",
                    "sources": [],
                    "error": "Failed to generate query embedding"
                }
            
            # Step 2: Retrieve relevant document chunks
            relevant_chunks = self.db_manager.similarity_search(
                query_embedding=query_embedding,
                threshold=Config.SIMILARITY_THRESHOLD,
                limit=max_results
            )
            
            if not relevant_chunks:
                return {
                    "answer": "I don't have any relevant information to answer this question based on the available contract documents.",
                    "sources": [],
                    "retrieved_chunks": 0
                }
            
            # Step 3: Prepare context and generate answer
            context, sources = self._prepare_context_and_sources(relevant_chunks)
            
            # Step 4: Generate response using LLM
            response = self._generate_response(question, context)
            
            logger.info(f"RAG query completed successfully with {len(relevant_chunks)} chunks retrieved")
            
            return {
                "answer": response,
                "sources": sources,
                "retrieved_chunks": len(relevant_chunks),
                "similarity_scores": [chunk.get("similarity", 0) for chunk in relevant_chunks]
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": [],
                "error": str(e)
            }
    
    def _prepare_context_and_sources(self, chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Prepare context string and source information from retrieved chunks"""
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(chunks):
            # Get document information
            document = self.db_manager.get_document_by_id(chunk["document_id"])
            if document:
                # Format context with source attribution
                context_part = f"[Source {i+1}: {document['filename']}]\n{chunk['chunk_text']}\n"
                context_parts.append(context_part)
                
                # Add source information
                sources.append({
                    "id": i + 1,
                    "filename": document['filename'],
                    "document_id": chunk['document_id'],
                    "similarity": round(chunk.get('similarity', 0), 3),
                    "chunk_text": chunk['chunk_text'][:200] + "..." if len(chunk['chunk_text']) > 200 else chunk['chunk_text']
                })
        
        context = "\n\n".join(context_parts)
        return context, sources
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using the LLM with conversation memory"""
        try:
            # Create the chain with memory
            chain = self.prompt_template | self.llm | StrOutputParser()
            
            # Generate response with context and history
            response = chain.invoke({
                "context": context, 
                "question": question,
                "chat_history": list(self.chat_history)
            })
            
            # Save to memory
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response))
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return "I encountered an error while generating the response. Please try again."
    
    def update_memory(self, user_message: str, ai_response: str):
        """Update conversation memory manually"""
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=ai_response))
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.chat_history.clear()
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search without LLM generation
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_query_embedding(query)
            if not query_embedding:
                return []
            
            # Retrieve similar chunks
            chunks = self.db_manager.similarity_search(
                query_embedding=query_embedding,
                threshold=Config.SIMILARITY_THRESHOLD,
                limit=limit
            )
            
            # Enrich with document information
            enriched_results = []
            for chunk in chunks:
                document = self.db_manager.get_document_by_id(chunk["document_id"])
                if document:
                    enriched_results.append({
                        "chunk_text": chunk["chunk_text"],
                        "similarity": chunk.get("similarity", 0),
                        "document_id": chunk["document_id"],
                        "filename": document["filename"],
                        "file_type": document["file_type"],
                        "document_metadata": document.get("metadata", {})
                    })
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def get_contract_insights(self, contract_query: str) -> Dict[str, Any]:
        """
        Get specific insights about contracts
        
        Args:
            contract_query: Query about contract details
            
        Returns:
            Structured insights about contracts
        """
        try:
            # Use specialized prompt for contract insights
            insight_prompt = ChatPromptTemplate.from_template("""
You are a contract analysis expert. Analyze the provided contract information and extract key insights.

Contract Information:
{context}

Query: {question}

Please provide insights in the following structure:
1. Key Facts: List important facts mentioned
2. Dates: Any relevant dates (start, end, renewal)
3. Parties: Companies or entities involved
4. Financial Terms: Amounts, costs, or financial obligations
5. Important Clauses: Key contractual provisions
6. Potential Issues: Any concerns or conflicts mentioned

Analysis:
            """)
            
            # Generate query embedding and retrieve context
            query_embedding = self.embedding_manager.generate_query_embedding(contract_query)
            if not query_embedding:
                return {"error": "Could not process query"}
            
            chunks = self.db_manager.similarity_search(
                query_embedding=query_embedding,
                threshold=0.6,  # Lower threshold for broader context
                limit=8
            )
            
            if not chunks:
                return {"insights": "No relevant contract information found"}
            
            context, sources = self._prepare_context_and_sources(chunks)
            
            # Generate insights
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | insight_prompt
                | self.llm
                | StrOutputParser()
            )
            
            insights = chain.invoke({"context": context, "question": contract_query})
            
            return {
                "insights": insights,
                "sources": sources,
                "analyzed_chunks": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Contract insights generation failed: {e}")
            return {"error": f"Failed to generate insights: {str(e)}"}
    
    def find_similar_contracts(self, reference_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find contracts similar to the reference text
        
        Args:
            reference_text: Text to find similar contracts for
            limit: Maximum number of similar contracts to return
            
        Returns:
            List of similar contracts with metadata
        """
        try:
            # Generate embedding for reference text
            reference_embedding = self.embedding_manager.generate_query_embedding(reference_text)
            if not reference_embedding:
                return []
            
            # Find similar document chunks
            similar_chunks = self.db_manager.similarity_search(
                query_embedding=reference_embedding,
                threshold=0.7,  # Higher threshold for similarity
                limit=limit * 2  # Get more chunks to find unique documents
            )
            
            # Group by document and get unique similar documents
            seen_documents = set()
            similar_documents = []
            
            for chunk in similar_chunks:
                doc_id = chunk["document_id"]
                if doc_id not in seen_documents:
                    document = self.db_manager.get_document_by_id(doc_id)
                    if document:
                        similar_documents.append({
                            "document_id": doc_id,
                            "filename": document["filename"],
                            "similarity": chunk.get("similarity", 0),
                            "file_type": document["file_type"],
                            "relevant_excerpt": chunk["chunk_text"][:300] + "..."
                        })
                        seen_documents.add(doc_id)
                        
                        if len(similar_documents) >= limit:
                            break
            
            return similar_documents
            
        except Exception as e:
            logger.error(f"Similar contract search failed: {e}")
            return []