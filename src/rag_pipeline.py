"""
RAG (Retrieval-Augmented Generation) pipeline for CLM automation system.
Implements document retrieval and question answering using LangChain.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
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

            # Step 2: Attempt to enhance the query for broader recall
            enhanced_query = self._enhance_query(question)
            if enhanced_query != question:
                enhanced_embedding = self.embedding_manager.generate_query_embedding(enhanced_query)
                if enhanced_embedding:
                    query_embedding = enhanced_embedding

            # Step 3: Retrieve relevant document chunks with adaptive thresholds
            initial_threshold = self._determine_similarity_threshold(question)
            thresholds = self._build_threshold_schedule(initial_threshold)
            raw_chunks = self._search_with_fallback(
                query_embedding=query_embedding,
                thresholds=thresholds,
                limit=max_results * 2
            )

            if raw_chunks:
                raw_chunks = self._maybe_augment_with_direct_lookup(
                    question,
                    raw_chunks,
                    max_results * 3
                )
            else:
                raw_chunks = self._direct_text_lookup(question, max_results * 3)

            if not raw_chunks:
                return {
                    "answer": "I don't have any relevant information to answer this question based on the available contract documents.",
                    "sources": [],
                    "retrieved_chunks": 0
                }

            # Step 4: Rerank chunks to prioritize the most relevant context for the question
            relevant_chunks = self._rerank_chunks(raw_chunks, question, max_results)

            # Step 5: Prepare context and generate answer
            context, sources = self._prepare_context_and_sources(relevant_chunks)

            # Step 6: Generate response using LLM
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
        documents_cache: Dict[str, Optional[Dict[str, Any]]] = {}

        for i, chunk in enumerate(chunks):
            doc_id = chunk.get("document_id")
            if not doc_id:
                continue

            if doc_id not in documents_cache:
                documents_cache[doc_id] = self.db_manager.get_document_by_id(doc_id)

            document = documents_cache.get(doc_id)
            if not document:
                continue

            chunk_text = chunk.get('chunk_text') or ''

            # Format context with source attribution
            context_part = f"[Source {i+1}: {document['filename']}]\n{chunk_text}\n"
            context_parts.append(context_part)

            # Add source information
            formatted_text = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            sources.append({
                "id": i + 1,
                "filename": document['filename'],
                "document_id": doc_id,
                "similarity": round(chunk.get('similarity', 0), 3),
                "chunk_text": formatted_text
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
            
            thresholds = self._build_threshold_schedule(Config.SIMILARITY_THRESHOLD)
            raw_chunks = self._search_with_fallback(
                query_embedding=query_embedding,
                thresholds=thresholds,
                limit=max(limit * 2, 5)
            )

            if raw_chunks:
                raw_chunks = self._maybe_augment_with_direct_lookup(
                    query,
                    raw_chunks,
                    limit * 3
                )
            else:
                raw_chunks = self._direct_text_lookup(query, limit * 3)

            if not raw_chunks:
                return []

            ranked_chunks = self._rerank_chunks(raw_chunks, query, limit)

            # Enrich with document information
            enriched_results = []
            documents_cache: Dict[str, Optional[Dict[str, Any]]] = {}
            for chunk in ranked_chunks:
                doc_id = chunk.get("document_id")
                if not doc_id:
                    continue

                if doc_id not in documents_cache:
                    documents_cache[doc_id] = self.db_manager.get_document_by_id(doc_id)

                document = documents_cache.get(doc_id)
                if not document:
                    continue

                enriched_results.append({
                    "chunk_text": chunk.get("chunk_text") or "",
                    "similarity": chunk.get("similarity", 0),
                    "document_id": doc_id,
                    "filename": document.get("filename"),
                    "file_type": document.get("file_type"),
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
            
            thresholds = self._build_threshold_schedule(0.6)
            raw_chunks = self._search_with_fallback(
                query_embedding=query_embedding,
                thresholds=thresholds,
                limit=12
            )

            if raw_chunks:
                raw_chunks = self._maybe_augment_with_direct_lookup(
                    contract_query,
                    raw_chunks,
                    18
                )
            else:
                raw_chunks = self._direct_text_lookup(contract_query, 16)

            if not raw_chunks:
                return {"insights": "No relevant contract information found"}

            insight_chunks = self._rerank_chunks(raw_chunks, contract_query, 8)

            context, sources = self._prepare_context_and_sources(insight_chunks)

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
                "analyzed_chunks": len(insight_chunks)
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
            
            thresholds = self._build_threshold_schedule(0.7)
            similar_chunks = self._search_with_fallback(
                query_embedding=reference_embedding,
                thresholds=thresholds,
                limit=max(limit * 3, 6)
            )

            if similar_chunks:
                similar_chunks = self._maybe_augment_with_direct_lookup(
                    reference_text,
                    similar_chunks,
                    max(limit * 4, 8)
                )
            else:
                similar_chunks = self._direct_text_lookup(reference_text, max(limit * 3, 6))

            # Group by document and get unique similar documents
            seen_documents = set()
            similar_documents = []

            for chunk in similar_chunks:
                doc_id = chunk.get("document_id")
                if doc_id not in seen_documents:
                    document = self.db_manager.get_document_by_id(doc_id)
                    if document:
                        chunk_text = chunk.get("chunk_text") or ""
                        excerpt = chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text
                        similar_documents.append({
                            "document_id": doc_id,
                            "filename": document.get("filename"),
                            "similarity": chunk.get("similarity", 0),
                            "file_type": document.get("file_type"),
                            "relevant_excerpt": excerpt
                        })
                        seen_documents.add(doc_id)
                        
                        if len(similar_documents) >= limit:
                            break
            
            return similar_documents
            
        except Exception as e:
            logger.error(f"Similar contract search failed: {e}")
            return []

    def _enhance_query(self, question: str) -> str:
        """Enhance the query to improve retrieval relevance"""
        contract_terms = [
            "contract", "agreement", "terms", "conditions", "clause",
            "provision", "obligation", "party", "legal", "document"
        ]

        aspect_keywords = {
            "expir": ["expiration", "renewal", "term", "duration"],
            "payment": ["financial", "cost", "price", "amount", "fee"],
            "party": ["company", "organization", "entity", "vendor"],
            "risk": ["liability", "responsibility", "indemnity", "insurance"],
            "termination": ["end", "cancel", "breach", "violation"]
        }

        question_lower = question.lower()
        enhanced_terms: List[str] = []

        for aspect, keywords in aspect_keywords.items():
            if aspect in question_lower:
                enhanced_terms.extend(keywords[:2])

        if enhanced_terms:
            enhanced_query = f"{question} {' '.join(enhanced_terms)}"
        else:
            enhanced_query = f"{question} {' '.join(contract_terms[:3])}"

        logger.debug("Enhanced query: '%s' -> '%s'", question, enhanced_query)
        return enhanced_query

    def _determine_similarity_threshold(self, question: str) -> float:
        """Determine appropriate similarity threshold based on query type"""
        question_lower = question.lower()

        specific_indicators = [
            "specific", "exact", "particular", "named", "called",
            "company name", "contract name", "agreement with"
        ]

        general_indicators = [
            "overview", "summary", "general", "all", "any",
            "what are", "tell me about", "explain"
        ]

        if any(indicator in question_lower for indicator in specific_indicators):
            return 0.8
        if any(indicator in question_lower for indicator in general_indicators):
            return 0.65
        return Config.SIMILARITY_THRESHOLD

    def _build_threshold_schedule(self, initial_threshold: float) -> List[float]:
        """Create a descending list of thresholds for fallback retrieval"""
        candidate_values = [initial_threshold, 0.75, 0.65, 0.55, 0.45, 0.0]
        thresholds: List[float] = []

        for value in candidate_values:
            if value is None:
                continue
            clamped = max(min(value, 0.95), 0.0)
            if not thresholds or abs(thresholds[-1] - clamped) > 1e-3:
                thresholds.append(clamped)

        return thresholds

    def _search_with_fallback(self, query_embedding: List[float], thresholds: List[float], limit: int) -> List[Dict[str, Any]]:
        """Run similarity search with fallback thresholds until results are found"""
        for attempt, threshold in enumerate(thresholds, start=1):
            chunks = self.db_manager.similarity_search(
                query_embedding=query_embedding,
                threshold=threshold,
                limit=limit
            )
            if chunks:
                logger.debug(
                    "Similarity search succeeded on attempt %s with threshold %.2f and %s chunks",
                    attempt,
                    threshold,
                    len(chunks)
                )
                return chunks

        logger.debug("Similarity search returned no chunks after %s attempts", len(thresholds))
        return []

    def _direct_text_lookup(self, question: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback to direct text search against chunk contents"""
        candidates = self._generate_text_search_candidates(question)
        if not candidates:
            return []

        logger.debug("Text search candidates: %s", candidates)

        combined: List[Dict[str, Any]] = []
        seen_pairs = set()

        for phrase in candidates:
            remaining = max(limit - len(combined), 1)
            results = self.db_manager.search_chunks_by_text(phrase, limit=remaining)
            if not results:
                continue

            for item in results:
                doc_id = item.get('document_id')
                chunk_index = item.get('chunk_index')
                key = (doc_id, chunk_index)
                if key in seen_pairs:
                    continue

                combined.append({
                    **item,
                    'similarity': item.get('similarity', 0.82),
                    'relevance_score': 0.82
                })
                seen_pairs.add(key)

                if len(combined) >= limit:
                    break

            if len(combined) >= limit:
                break

        if combined:
            logger.debug("Text search returned %s unique chunks", len(combined))

        return combined

    def _maybe_augment_with_direct_lookup(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Optionally blend direct text search hits with similarity results"""
        if not self._should_use_direct_lookup(question):
            return chunks

        direct_hits = self._direct_text_lookup(question, limit)
        if not direct_hits:
            return chunks

        combined: List[Dict[str, Any]] = []
        seen_pairs = set()

        for item in chunks + direct_hits:
            doc_id = item.get('document_id')
            chunk_index = item.get('chunk_index')
            key = (doc_id, chunk_index)
            if key in seen_pairs:
                continue
            combined.append(item)
            seen_pairs.add(key)
            if len(combined) >= limit:
                break

        logger.info(
            "Augmented similarity results with %s direct text hits (total %s)",
            len(direct_hits),
            len(combined)
        )
        return combined

    def _should_use_direct_lookup(self, question: str) -> bool:
        """Heuristic to decide when to blend in direct text search results"""
        normalized = question.strip()
        if not normalized:
            return False

        if '\n' in normalized:
            return True

        if len(normalized) >= 120:
            return True

        clause_markers = ['shall be entitled', 'section', 'clause', 'fsow', 'sow']
        normalized_lower = normalized.lower()
        return any(marker in normalized_lower for marker in clause_markers)

    def _generate_text_search_candidates(self, question: str) -> List[str]:
        """Extract candidate phrases for direct text lookup"""
        # Normalize whitespace and strip markup
        compact = ' '.join(question.split())
        if not compact:
            return []

        candidates: List[str] = []

        # Prefer chunks from individual lines (useful for pasted clauses)
        for line in question.splitlines():
            cleaned = self._clean_candidate_phrase(line)
            if cleaned:
                candidates.append(cleaned)

        if not candidates:
            candidates.append(self._clean_candidate_phrase(compact))

        # Add sliding windows to improve partial matches
        tokens = compact.split()
        if len(tokens) >= 4:
            window_size = min(8, max(len(tokens) // 2, 4))
            step = max(window_size // 2, 1)

            max_start = max(len(tokens) - window_size + 1, 1)
            for start in range(0, max_start, step):
                segment = ' '.join(tokens[start:start + window_size])
                cleaned = self._clean_candidate_phrase(segment)
                if cleaned:
                    candidates.append(cleaned)

            # Add shorter n-grams to catch partial overlaps
            short_window = 5
            for start in range(0, max(len(tokens) - short_window + 1, 1)):
                segment = ' '.join(tokens[start:start + short_window])
                cleaned = self._clean_candidate_phrase(segment)
                if cleaned:
                    candidates.append(cleaned)

        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for phrase in candidates:
            if phrase not in seen:
                seen.add(phrase)
                unique_candidates.append(phrase)

        return unique_candidates[:20]

    def _clean_candidate_phrase(self, text: str) -> str:
        """Normalize a candidate phrase for SQL ILIKE search"""
        stripped = text.strip().strip('?"')
        if len(stripped) < 8:
            return ''

        # Remove repeated whitespace and dangling punctuation
        stripped = re.sub(r'\s+', ' ', stripped)
        strip_chars = ".,;:?!\"' )("
        stripped = stripped.strip(strip_chars)

        # Remove simple list prefixes (e.g., "c)", "-", "•")
        stripped = re.sub(r'^[\-•*]+\s*', '', stripped)
        stripped = re.sub(r'^[A-Za-z]\)\s*', '', stripped)

        if len(stripped) < 6:
            return ''

        return stripped

    def _rerank_chunks(self, chunks: List[Dict[str, Any]], question: str, limit: int) -> List[Dict[str, Any]]:
        """Re-rank document chunks based on relevance to the question"""
        if not chunks:
            return []

        question_lower = question.lower()
        question_tokens = set(question_lower.split())

        scored_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            score = chunk.get('similarity', 0)
            chunk_text = chunk.get('chunk_text') or ''
            chunk_text_lower = chunk_text.lower()
            chunk_tokens = set(chunk_text_lower.split())

            token_overlap = len(question_tokens.intersection(chunk_tokens))
            overlap_bonus = min(token_overlap * 0.05, 0.2)

            contract_indicators = [
                'agreement', 'contract', 'party', 'clause', 'provision',
                'terms', 'conditions', 'obligations', 'rights'
            ]
            contract_bonus = sum(0.02 for term in contract_indicators if term in chunk_text_lower)
            contract_bonus = min(contract_bonus, 0.15)

            type_bonus = 0.0
            if 'expir' in question_lower and any(term in chunk_text_lower for term in ['expir', 'renew', 'term', 'end']):
                type_bonus = 0.1
            elif 'payment' in question_lower and any(term in chunk_text_lower for term in ['payment', 'cost', 'fee', 'amount']):
                type_bonus = 0.1
            elif 'party' in question_lower and any(term in chunk_text_lower for term in ['company', 'corporation', 'entity']):
                type_bonus = 0.1

            final_score = score + overlap_bonus + contract_bonus + type_bonus

            scored_chunks.append({
                **chunk,
                'relevance_score': final_score,
                'token_overlap': token_overlap,
                'contract_relevance': contract_bonus
            })

        scored_chunks.sort(key=lambda item: item.get('relevance_score', 0), reverse=True)

        min_relevance = 0.6
        filtered_chunks = [chunk for chunk in scored_chunks if chunk.get('relevance_score', 0) >= min_relevance]

        if limit <= 0:
            return filtered_chunks or scored_chunks

        return (filtered_chunks[:limit] if filtered_chunks else scored_chunks[:limit])
