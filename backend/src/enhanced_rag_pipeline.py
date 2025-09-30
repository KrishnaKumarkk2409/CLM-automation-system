"""
Enhanced RAG Pipeline for CLM automation system.
Integrates hybrid search, advanced re-ranking, smart chunking, and specialized embeddings.
"""

import logging
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from collections import deque
import time
from datetime import datetime

from src.config import Config
from src.database import DatabaseManager
from src.embeddings import EmbeddingManager
from backend.src.hybrid_search import HybridSearchEngine, SearchResult, DocumentSimilarityResult
from backend.src.reranker import AdvancedReranker, RankedResult
from backend.src.enhanced_embeddings import EnhancedEmbeddingManager
from backend.src.smart_chunker import SmartDocumentChunker
from src.similarity_search import HybridSimilarityEngine

logger = logging.getLogger(__name__)

class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with advanced search and retrieval capabilities"""

    def __init__(self, db_manager: DatabaseManager, embedding_manager: EmbeddingManager = None):
        self.db_manager = db_manager

        # Use enhanced embedding manager if available, fallback to original
        if embedding_manager:
            self.embedding_manager = embedding_manager
        else:
            self.embedding_manager = EmbeddingManager(db_manager)
        self.enhanced_embedding_manager = EnhancedEmbeddingManager(db_manager)

        # Initialize NEW hybrid similarity engine (cosine + dot product)
        self.hybrid_similarity = HybridSimilarityEngine(
            db_manager=db_manager,
            embedding_manager=self.embedding_manager,
            candidate_pool=50,
            cosine_weight=0.7
        )

        # Keep old hybrid search for backward compatibility (TF-IDF based)
        self.hybrid_search = HybridSearchEngine(db_manager, self.enhanced_embedding_manager)
        self.reranker = AdvancedReranker(db_manager)
        self.smart_chunker = SmartDocumentChunker()

        # Initialize conversation memory
        self.chat_history = deque(maxlen=10)

        # Initialize OpenAI chat model
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=Config.OPENAI_API_KEY
        )

        # Enhanced prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
You are an advanced legal assistant specializing in contract analysis with enhanced document understanding capabilities.

Your enhanced capabilities include:
- Multi-factor relevance analysis
- Temporal awareness for date-sensitive queries
- Legal term specialization
- Entity recognition and matching
- Contextual understanding across document sections

Analyze the user's question to understand:
1. Query intent (expiration, financial, legal terms, parties, document search)
2. Temporal requirements (current, future, past events)
3. Entity focus (companies, people, amounts, dates)
4. Legal domain specificity
5. Relationship to conversation history

Provide comprehensive answers using the enhanced context, prioritizing:
- High-relevance information based on multi-factor scoring
- Temporally relevant content for time-sensitive queries
- Entity-matched information for specific searches
- Legal term precision for compliance queries

Context from enhanced document retrieval:
{context}

Enhanced search metadata:
- Query type: {query_type}
- Search method: {search_method}
- Reranking applied: {reranking_applied}
- Results confidence: {confidence_score}

Instructions:
- Leverage the enhanced search metadata to provide targeted responses
- Reference specific document sections and legal provisions
- Highlight temporal relevance when applicable
- Explain entity relationships when identified
- Provide actionable insights based on legal analysis
- If context is insufficient, specify what additional information would be helpful
- Always cite sources with relevance confidence scores
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # Performance tracking
        self.performance_metrics = {
            'total_queries': 0,
            'average_response_time': 0.0,
            'hybrid_search_usage': 0,
            'reranking_improvements': 0,
            'cache_hit_rate': 0.0
        }

        logger.info("Enhanced RAG pipeline initialized with advanced components")

    def query(self, question: str, max_results: int = 10,
              use_hybrid_search: bool = True, use_reranking: bool = False,
              user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process query using enhanced RAG pipeline with advanced search and ranking

        Args:
            question: User's question
            max_results: Maximum number of results to return
            use_hybrid_search: Whether to use hybrid search (dense + sparse)
            use_reranking: Whether to apply advanced re-ranking
            user_context: Optional user context for personalization

        Returns:
            Enhanced response with detailed metadata and explanations
        """
        start_time = time.time()

        try:
            logger.info(f"Processing enhanced RAG query: {question[:100]}...")

            # Step 1: Enhanced query processing and search
            if use_hybrid_search:
                search_results = self._hybrid_search_retrieval(question, max_results * 2)
                search_method = "hybrid"
                self.performance_metrics['hybrid_search_usage'] += 1
            else:
                search_results = self._traditional_search_retrieval(question, max_results * 2)
                search_method = "traditional"

            if not search_results:
                return self._create_no_results_response(question)

            # Step 2: Advanced re-ranking (disabled by default for performance)
            if use_reranking:
                try:
                    logger.info(f"Applying reranking to {len(search_results)} search results")
                    ranked_results = self.reranker.rerank_results(
                        question,
                        self._convert_search_results(search_results),
                        user_context
                    )

                    # Measure re-ranking improvement
                    if ranked_results:
                        original_avg_score = sum(r.original_score for r in ranked_results) / len(ranked_results)
                        reranked_avg_score = sum(r.reranked_score for r in ranked_results) / len(ranked_results)

                        if reranked_avg_score > original_avg_score:
                            self.performance_metrics['reranking_improvements'] += 1

                        reranking_applied = True
                        final_results = ranked_results[:max_results]
                        logger.info(f"Reranking completed successfully")
                    else:
                        logger.warning("Reranking returned no results, falling back to original")
                        final_results = self._convert_to_ranked_results(search_results[:max_results])
                        reranking_applied = False
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}, using original search results")
                    final_results = self._convert_to_ranked_results(search_results[:max_results])
                    reranking_applied = False
            else:
                final_results = self._convert_to_ranked_results(search_results[:max_results])
                reranking_applied = False

            # Step 3: Context preparation with enhanced metadata
            context, sources, metadata = self._prepare_enhanced_context(final_results, question)

            # Step 4: Query classification for enhanced prompting
            query_type = self._classify_enhanced_query(question)

            # Step 5: Generate enhanced response
            response = self._generate_enhanced_response(
                question, context, query_type, search_method,
                reranking_applied, metadata
            )

            # Step 6: Calculate performance metrics
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time)

            # Step 7: Prepare final response
            enhanced_response = {
                "answer": response,
                "sources": sources,
                "metadata": {
                    "query_type": query_type,
                    "search_method": search_method,
                    "reranking_applied": reranking_applied,
                    "results_count": len(final_results),
                    "response_time": response_time,
                    "confidence_score": self._calculate_confidence_score(final_results),
                    **metadata
                },
                "ranking_explanations": [r.ranking_explanation for r in final_results] if reranking_applied else [],
                "performance_insights": self._get_performance_insights(final_results)
            }

            logger.info(f"Enhanced RAG query completed in {response_time:.2f}s")
            return enhanced_response

        except Exception as e:
            logger.exception("Enhanced RAG query failed")
            return self._create_error_response(str(e))

    def _hybrid_search_retrieval(self, question: str, limit: int) -> List[SearchResult]:
        """Perform hybrid search retrieval"""
        try:
            return self.hybrid_search.search(question, limit)
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def _traditional_search_retrieval(self, question: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback to traditional search retrieval"""
        try:
            # Generate enhanced query embedding
            query_embedding = self.enhanced_embedding_manager.generate_query_embedding(question)

            if not query_embedding:
                return []

            # Traditional similarity search
            similar_chunks = self.db_manager.similarity_search(
                query_embedding=query_embedding,
                threshold=0.6,
                limit=limit
            )

            return similar_chunks

        except Exception as e:
            logger.error(f"Traditional search failed: {e}")
            return []

    def _convert_search_results(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Convert SearchResult objects to dictionary format for reranker"""
        converted_results = []

        for result in search_results:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            converted_result = {
                'document_id': result.document_id,
                'chunk_text': result.chunk_text,
                'similarity': result.hybrid_score,
                'metadata': {
                    'dense_score': result.dense_score,
                    'sparse_score': result.sparse_score,
                    'chunk_index': result.chunk_index,
                    **metadata
                }
            }
            converted_results.append(converted_result)

        return converted_results

    def _convert_to_ranked_results(self, search_results) -> List[RankedResult]:
        """Convert search results to RankedResult format without re-ranking"""
        from src.reranker import RankedResult, RerankingFeatures

        ranked_results = []

        for result in search_results:
            # Handle both SearchResult objects and dictionaries
            if hasattr(result, 'hybrid_score'):
                # It's a SearchResult object from hybrid_search
                similarity = result.hybrid_score
                document_id = result.document_id
                chunk_text = result.chunk_text
                metadata = result.metadata if isinstance(result.metadata, dict) else {}
            else:
                # It's a dictionary
                similarity = result.get('similarity', 0.0)
                document_id = result.get('document_id', '')
                chunk_text = result.get('chunk_text', '')
                metadata = result.get('metadata', {})

            # Create minimal features
            features = RerankingFeatures(
                semantic_similarity=similarity,
                keyword_match_score=0.5,
                document_relevance=0.5,
                temporal_relevance=0.5,
                structural_importance=0.5,
                entity_match_score=0.5,
                legal_term_relevance=0.5,
                user_feedback_score=0.5
            )

            ranked_result = RankedResult(
                document_id=document_id,
                chunk_text=chunk_text,
                original_score=similarity,
                reranked_score=similarity,
                features=features,
                chunk_metadata=metadata,
                ranking_explanation="Hybrid similarity-based ranking (cosine + sparse)"
            )

            ranked_results.append(ranked_result)

        return ranked_results

    def _prepare_enhanced_context(self, results: List[RankedResult], question: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Prepare enhanced context with detailed metadata"""
        context_parts = []
        sources = []

        # Extract metadata for analysis
        total_confidence = 0.0
        relevance_distribution = {}
        temporal_relevance_count = 0

        for i, result in enumerate(results):
            # Get document information
            document = self.db_manager.get_document_by_id(result.document_id)
            if document:
                chunk_metadata = result.chunk_metadata if isinstance(result.chunk_metadata, dict) else {}
                section_title = chunk_metadata.get('section_title') or 'General'
                chunk_type = chunk_metadata.get('chunk_type')
                chunk_text = result.chunk_text or ""
                features = result.features

                # Enhanced context formatting with metadata
                confidence_indicator = "🔥" if result.reranked_score > 0.8 else "✓" if result.reranked_score > 0.6 else "~"

                context_part = f"""
[Source {i+1} {confidence_indicator} Relevance: {result.reranked_score:.1%}]
Document: {document['filename']}
Section: {section_title}
Content: {chunk_text}
---
"""
                context_parts.append(context_part)

                # Enhanced source information
                source_info = {
                    "id": i + 1,
                    "filename": document['filename'],
                    "document_id": result.document_id,
                    "relevance_score": round(result.reranked_score, 3),
                    "original_score": round(result.original_score, 3),
                    "chunk_text": (chunk_text[:200] + "...") if len(chunk_text) > 200 else chunk_text,
                    "section_title": chunk_metadata.get('section_title'),
                    "chunk_type": chunk_type,
                    "features": {
                        "semantic_similarity": round(features.semantic_similarity, 3),
                        "keyword_match": round(features.keyword_match_score, 3),
                        "legal_term_relevance": round(features.legal_term_relevance, 3),
                        "temporal_relevance": round(features.temporal_relevance, 3),
                        "entity_match": round(features.entity_match_score, 3)
                    },
                    "ranking_explanation": result.ranking_explanation
                }
                sources.append(source_info)

                # Metadata analysis
                total_confidence += result.reranked_score

                distribution_key = chunk_type or 'unknown'
                relevance_distribution[distribution_key] = relevance_distribution.get(distribution_key, 0) + 1

                if features.temporal_relevance > 0.7:
                    temporal_relevance_count += 1

        context = "\n".join(context_parts)

        # Enhanced metadata
        metadata = {
            "average_confidence": total_confidence / len(results) if results else 0.0,
            "relevance_distribution": relevance_distribution,
            "temporal_relevance_count": temporal_relevance_count,
            "high_confidence_results": sum(1 for r in results if r.reranked_score > 0.8),
            "context_length": len(context),
            "source_count": len(sources)
        }

        return context, sources, metadata

    def _classify_enhanced_query(self, question: str) -> str:
        """Enhanced query classification with more granular types"""
        question_lower = question.lower()

        # Enhanced classification patterns
        classification_patterns = {
            'expiration_analysis': ['expir', 'renew', 'term', 'deadline', 'due date', 'end date'],
            'financial_inquiry': ['cost', 'price', 'payment', 'fee', 'amount', 'budget', '$', 'financial'],
            'party_investigation': ['company', 'vendor', 'supplier', 'client', 'partner', 'entity', 'party'],
            'legal_compliance': ['clause', 'provision', 'term', 'condition', 'liability', 'breach', 'compliance'],
            'temporal_analysis': ['when', 'date', 'time', 'recent', 'current', 'upcoming', 'past', 'future'],
            'document_discovery': ['find', 'search', 'locate', 'show', 'list', 'document', 'contract'],
            'comparative_analysis': ['compare', 'similar', 'different', 'versus', 'between', 'contrast'],
            'risk_assessment': ['risk', 'liability', 'indemnif', 'insurance', 'penalty', 'damages'],
            'intellectual_property': ['ip', 'intellectual property', 'copyright', 'trademark', 'patent'],
            'general_inquiry': []  # Default
        }

        for query_type, patterns in classification_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                return query_type

        return 'general_inquiry'

    def _generate_enhanced_response(self, question: str, context: str, query_type: str,
                                  search_method: str, reranking_applied: bool,
                                  metadata: Dict[str, Any]) -> str:
        """Generate response using enhanced prompt template"""
        try:
            chain = self.prompt_template | self.llm | StrOutputParser()

            response = chain.invoke({
                "context": context,
                "question": question,
                "query_type": query_type,
                "search_method": search_method,
                "reranking_applied": reranking_applied,
                "confidence_score": f"{metadata.get('average_confidence', 0):.1%}",
                "chat_history": list(self.chat_history)
            })

            # Update conversation memory
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response))

            return response.strip()

        except Exception as e:
            logger.error(f"Enhanced response generation failed: {e}")
            return "I encountered an error while generating the enhanced response. Please try again."

    def _calculate_confidence_score(self, results: List[RankedResult]) -> float:
        """Calculate overall confidence score for the results"""
        if not results:
            return 0.0

        # Weighted confidence based on top results
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Diminishing weights for top 5 results
        total_weighted_score = 0.0
        total_weights = 0.0

        for i, result in enumerate(results[:5]):
            weight = weights[i] if i < len(weights) else 0.1
            total_weighted_score += result.reranked_score * weight
            total_weights += weight

        return total_weighted_score / total_weights if total_weights > 0 else 0.0

    def _get_performance_insights(self, results: List[RankedResult]) -> Dict[str, Any]:
        """Get performance insights from the results"""
        if not results:
            return {}

        # Analyze result quality
        score_improvements = []
        feature_analysis = {}

        for result in results:
            if result.original_score > 0:
                improvement = (result.reranked_score - result.original_score) / result.original_score
                score_improvements.append(improvement)

            # Aggregate feature scores
            features = result.features
            for feature_name in ['semantic_similarity', 'keyword_match_score', 'legal_term_relevance']:
                if feature_name not in feature_analysis:
                    feature_analysis[feature_name] = []
                feature_analysis[feature_name].append(getattr(features, feature_name))

        insights = {
            'average_score_improvement': sum(score_improvements) / len(score_improvements) if score_improvements else 0.0,
            'results_with_improvement': sum(1 for imp in score_improvements if imp > 0),
            'feature_averages': {
                feature: sum(scores) / len(scores)
                for feature, scores in feature_analysis.items()
            },
            'top_result_confidence': results[0].reranked_score if results else 0.0,
            'result_diversity': len(set(r.document_id for r in results))
        }

        return insights

    def _update_performance_metrics(self, response_time: float):
        """Update performance tracking metrics"""
        self.performance_metrics['total_queries'] += 1

        # Update average response time
        total_queries = self.performance_metrics['total_queries']
        current_avg = self.performance_metrics['average_response_time']
        new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
        self.performance_metrics['average_response_time'] = new_avg

        # Update cache hit rate from embedding manager
        embedding_stats = self.enhanced_embedding_manager.get_embedding_stats()
        self.performance_metrics['cache_hit_rate'] = embedding_stats.get('cache_hit_rate', 0.0)

    def _create_no_results_response(self, question: str) -> Dict[str, Any]:
        """Create response when no results are found"""
        return {
            "answer": "I don't have any relevant information to answer this question based on the available contract documents. Consider rephrasing your query or checking if the relevant documents have been processed.",
            "sources": [],
            "metadata": {
                "query_type": "no_results",
                "search_method": "none",
                "reranking_applied": False,
                "results_count": 0,
                "confidence_score": 0.0
            },
            "ranking_explanations": [],
            "performance_insights": {}
        }

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create response when an error occurs"""
        return {
            "answer": f"I encountered an error while processing your question: {error_message}",
            "sources": [],
            "metadata": {
                "query_type": "error",
                "search_method": "none",
                "reranking_applied": False,
                "results_count": 0,
                "confidence_score": 0.0,
                "error": error_message
            },
            "ranking_explanations": [],
            "performance_insights": {}
        }

    # Legacy compatibility methods
    def find_similar_contracts(self, reference_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find contracts similar to the provided reference text"""
        try:
            search_results = self.hybrid_search.search(reference_text, limit=max(limit * 3, 10))

            if not search_results:
                return []

            document_scores: Dict[str, float] = {}
            representative_chunks: Dict[str, str] = {}

            for result in search_results:
                doc_id = result.document_id
                score = result.hybrid_score

                if score <= 0:
                    continue

                if score > document_scores.get(doc_id, 0.0):
                    document_scores[doc_id] = score
                    representative_chunks[doc_id] = result.chunk_text

            ranked_documents = sorted(
                document_scores.items(),
                key=lambda item: item[1],
                reverse=True
            )

            similar_documents: List[Dict[str, Any]] = []

            for doc_id, score in ranked_documents:
                document = self.db_manager.get_document_by_id(doc_id)
                if not document:
                    continue

                similar_documents.append({
                    "document_id": doc_id,
                    "filename": document.get("filename", "Unknown document"),
                    "similarity": round(score, 4),
                    "file_type": document.get("file_type"),
                    "metadata": document.get("metadata", {}),
                    "excerpt": self._trim_excerpt(representative_chunks.get(doc_id, ""))
                })

                if len(similar_documents) >= limit:
                    break

            return similar_documents

        except Exception as e:
            logger.error(f"Enhanced similar contract search failed: {e}")
            return []

    def document_similarity(self, document_a: str, document_b: str,
                             aggregation: str = "mean", top_k: int = 3) -> Optional[Dict[str, Any]]:
        """Compute similarity between two documents"""
        try:
            similarity_result = self.hybrid_search.document_similarity(
                document_a=document_a,
                document_b=document_b,
                aggregation=aggregation,
                top_k=top_k
            )

            if not similarity_result:
                return None

            return self._serialize_similarity_result(similarity_result)

        except ValueError as e:
            logger.warning(f"Invalid similarity request: {e}")
            return None
        except Exception as e:
            logger.error(f"Document similarity computation failed: {e}")
            return None

    def compare_document_to_corpus(self, reference_document_id: str,
                                    limit: int = 5,
                                    aggregation: str = "mean",
                                    top_k: int = 3) -> List[Dict[str, Any]]:
        """Compare a reference document against other documents in the corpus"""
        try:
            candidates = self._get_candidate_document_ids(reference_document_id, limit=limit * 4)

            if not candidates:
                return []

            similarity_results = self.hybrid_search.compare_document_to_corpus(
                reference_document_id,
                candidates,
                aggregation=aggregation,
                top_k=top_k
            )

            serialized_results = [
                self._serialize_similarity_result(result)
                for result in similarity_results[:limit]
            ]

            return serialized_results

        except Exception as e:
            logger.error(f"Corpus similarity comparison failed: {e}")
            return []

    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Legacy compatibility method for semantic search"""
        results = self.query(query, max_results=limit, use_hybrid_search=False, use_reranking=False)

        # Convert to legacy format
        legacy_results = []
        for source in results.get('sources', []):
            legacy_result = {
                "chunk_text": source.get('chunk_text', ''),
                "similarity": source.get('relevance_score', 0.0),
                "document_id": source.get('document_id', ''),
                "filename": source.get('filename', ''),
                "file_type": "contract",  # Default
                "document_metadata": {}
            }
            legacy_results.append(legacy_result)

        return legacy_results

    def clear_memory(self):
        """Clear conversation memory"""
        self.chat_history.clear()

    def update_memory(self, user_message: str, ai_response: str):
        """Update conversation memory manually"""
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=ai_response))

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        base_metrics = self.performance_metrics.copy()

        # Add component-specific metrics
        hybrid_search_stats = self.hybrid_search.get_search_stats()
        embedding_stats = self.enhanced_embedding_manager.get_embedding_stats()

        return {
            **base_metrics,
            'hybrid_search_stats': hybrid_search_stats,
            'embedding_stats': embedding_stats,
            'components_status': {
                'hybrid_search': 'active',
                'reranker': 'active',
                'enhanced_embeddings': 'active',
                'smart_chunker': 'active'
            }
        }

    def optimize_performance(self, feedback_data: List[Dict[str, Any]] = None):
        """Optimize pipeline performance based on feedback"""
        logger.info("Optimizing enhanced RAG pipeline performance...")

        # Update search index if needed
        try:
            self.hybrid_search.update_index()
            logger.info("Search index updated")
        except Exception as e:
            logger.error(f"Failed to update search index: {e}")

        # Clear embedding cache periodically
        try:
            embedding_stats = self.enhanced_embedding_manager.get_embedding_stats()
            if embedding_stats.get('cache_size', 0) > 5000:
                self.enhanced_embedding_manager.clear_cache()
                logger.info("Embedding cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear embedding cache: {e}")

        logger.info("Performance optimization completed")

    # ------------------------------------------------------------------
    # Internal helpers for similarity reporting and formatting
    # ------------------------------------------------------------------

    def _trim_excerpt(self, text: str, max_chars: int = 280) -> str:
        """Create a trimmed excerpt for similarity previews"""
        if not text:
            return ""
        normalized = ' '.join(text.split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip() + '...'

    def _get_candidate_document_ids(self, reference_document_id: str, limit: int = 20) -> List[str]:
        """Fetch candidate document IDs excluding the reference document"""
        try:
            result = (
                self.db_manager.client
                .table('documents')
                .select('id')
                .neq('id', reference_document_id)
                .limit(max(limit, 1))
                .execute()
            )
            return [item['id'] for item in (result.data or []) if item.get('id')]
        except Exception as e:
            logger.error(f"Failed to fetch candidate document ids: {e}")
            return []

    def _serialize_similarity_result(self, result: DocumentSimilarityResult) -> Dict[str, Any]:
        """Serialize similarity result with associated document metadata"""
        payload = asdict(result)

        document_a = self.db_manager.get_document_by_id(result.document_a)
        document_b = self.db_manager.get_document_by_id(result.document_b)

        payload['documents'] = {
            'document_a': {
                'id': result.document_a,
                'filename': (document_a or {}).get('filename'),
                'metadata': (document_a or {}).get('metadata', {})
            },
            'document_b': {
                'id': result.document_b,
                'filename': (document_b or {}).get('filename'),
                'metadata': (document_b or {}).get('metadata', {})
            }
        }

        return payload
