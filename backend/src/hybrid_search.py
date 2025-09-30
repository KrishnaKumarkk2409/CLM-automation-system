"""
Hybrid Search Module for CLM automation system.
Combines dense vector search with sparse keyword matching for improved retrieval accuracy.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    _NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    nltk = None
    stopwords = None
    word_tokenize = None
    WordNetLemmatizer = None
    _NLTK_AVAILABLE = False

from src.config import Config
from src.database import DatabaseManager
from src.embeddings import EmbeddingManager

# Download required NLTK data when available
if _NLTK_AVAILABLE:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as exc:  # pragma: no cover - download best effort
        logging.getLogger(__name__).debug("NLTK resource download skipped: %s", exc)

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with comprehensive scoring"""
    document_id: str
    chunk_text: str
    dense_score: float
    sparse_score: float
    hybrid_score: float
    chunk_index: int
    metadata: Dict[str, Any]


@dataclass
class DocumentSimilarityResult:
    """Document-level similarity details"""
    document_a: str
    document_b: str
    similarity: float
    aggregation: str
    chunk_matches: List[Dict[str, Any]]
    stats: Dict[str, Any]

class QueryProcessor:
    """Advanced query processing with intent classification and expansion"""

    def __init__(self):
        if _NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self.tokenize: Callable[[str], List[str]] = word_tokenize
        else:
            # Fallback to lightweight processing without NLTK
            self.lemmatizer = None
            self.stop_words = set(ENGLISH_STOP_WORDS)
            self.tokenize = self._simple_tokenize

        # Legal/Contract domain synonyms and expansions
        self.legal_synonyms = {
            'agreement': ['contract', 'deal', 'arrangement', 'pact', 'accord'],
            'party': ['entity', 'organization', 'company', 'corporation', 'firm'],
            'termination': ['end', 'conclusion', 'expiry', 'cessation', 'cancellation'],
            'payment': ['compensation', 'remuneration', 'fee', 'amount', 'cost'],
            'liability': ['responsibility', 'obligation', 'accountability', 'debt'],
            'breach': ['violation', 'infringement', 'default', 'transgression'],
            'clause': ['provision', 'term', 'condition', 'section', 'article'],
            'renewal': ['extension', 'continuation', 'prolongation', 'restart'],
            'vendor': ['supplier', 'provider', 'contractor', 'service provider'],
            'intellectual property': ['IP', 'copyright', 'trademark', 'patent', 'trade secret']
        }

        # Query intent patterns
        self.intent_patterns = {
            'expiration_query': [
                r'expir\w+', r'renew\w+', r'end\w+', r'terminat\w+',
                r'due date', r'deadline', r'maturity'
            ],
            'financial_query': [
                r'cost\w*', r'price\w*', r'payment\w*', r'fee\w*',
                r'amount\w*', r'budget\w*', r'financial', r'\$[\d,]+'
            ],
            'party_query': [
                r'company', r'corporation', r'vendor', r'supplier',
                r'client', r'customer', r'partner', r'entity'
            ],
            'legal_terms_query': [
                r'clause\w*', r'provision\w*', r'term\w*', r'condition\w*',
                r'liability', r'breach', r'compliance', r'obligation'
            ],
            'document_search': [
                r'find\w*', r'search\w*', r'locate\w*', r'show\w*',
                r'document\w*', r'contract\w*', r'agreement\w*'
            ]
        }

        logger.info("Query processor initialized with legal domain knowledge")

    def classify_intent(self, query: str) -> List[str]:
        """Classify query intent based on patterns"""
        query_lower = query.lower()
        intents = []

        for intent, patterns in self.intent_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                intents.append(intent)

        return intents if intents else ['general_query']

    def expand_query(self, query: str, intents: List[str]) -> str:
        """Expand query with relevant synonyms and domain terms"""
        expanded_terms = []
        query_words = self.tokenize(query.lower())

        # Add original query
        expanded_terms.append(query)

        # Add synonyms for recognized terms
        for word in query_words:
            if word in self.legal_synonyms:
                # Add top 2 synonyms to avoid over-expansion
                expanded_terms.extend(self.legal_synonyms[word][:2])

        # Add intent-specific terms
        intent_expansions = {
            'expiration_query': ['expiration', 'renewal', 'term', 'duration'],
            'financial_query': ['financial', 'monetary', 'cost', 'pricing'],
            'party_query': ['contracting party', 'signatory', 'stakeholder'],
            'legal_terms_query': ['legal provision', 'contractual term', 'clause'],
            'document_search': ['contract document', 'agreement text']
        }

        for intent in intents:
            if intent in intent_expansions:
                expanded_terms.extend(intent_expansions[intent][:2])

        return ' '.join(expanded_terms)

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\$\%]', ' ', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        # Lemmatize words
        words = self.tokenize(text)

        if self.lemmatizer:
            processed = [
                self.lemmatizer.lemmatize(word)
                for word in words
                if word not in self.stop_words and len(word) > 2
            ]
        else:
            processed = [
                word for word in words
                if word not in self.stop_words and len(word) > 2
            ]

        return ' '.join(processed)

    def _simple_tokenize(self, text: str) -> List[str]:
        """Basic tokenizer when NLTK is unavailable"""
        return [token for token in re.split(r'[^a-z0-9]+', text.lower()) if token]

class HybridSearchEngine:
    """Hybrid search combining dense vector search with sparse keyword matching"""

    def __init__(self, db_manager: DatabaseManager, embedding_manager: EmbeddingManager):
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.query_processor = QueryProcessor()

        # Initialize TF-IDF for sparse search
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.document_chunks = []

        # Default weighting configuration
        self.default_dense_weight = 0.7
        self.default_sparse_weight = 0.3
        self.current_dense_weight = self.default_dense_weight
        self.current_sparse_weight = self.default_sparse_weight

        # Dense similarity threshold configuration
        base_threshold = Config.SIMILARITY_THRESHOLD if Config.SIMILARITY_THRESHOLD > 0 else 0.7
        self.default_dense_threshold = float(min(max(base_threshold, 0.05), 0.95))
        self.current_dense_threshold = self.default_dense_threshold

        # Initialize the search index
        self._build_search_index()

        logger.info("Hybrid search engine initialized")

    def _build_search_index(self):
        """Build TF-IDF index for sparse search"""
        try:
            # Get all document chunks
            chunks_result = self.db_manager.client.table('document_chunks')\
                .select('id, document_id, chunk_text, chunk_index, metadata')\
                .execute()

            if not chunks_result.data:
                logger.warning("No document chunks found for indexing")
                return

            self.document_chunks = chunks_result.data

            # Preprocess texts for TF-IDF
            processed_texts = []
            for chunk in self.document_chunks:
                processed_text = self.query_processor.preprocess_text(chunk['chunk_text'])
                processed_texts.append(processed_text)

            # Build TF-IDF matrix
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),  # Include bigrams and trigrams
                min_df=2,  # Ignore very rare terms
                max_df=0.8,  # Ignore very common terms
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            )

            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)

            logger.info(f"TF-IDF index built with {len(processed_texts)} documents")

        except Exception as e:
            logger.error(f"Failed to build search index: {e}")

    def search(self, query: str, limit: int = 10, dense_weight: float = None,
               sparse_weight: float = None) -> List[SearchResult]:
        """
        Perform hybrid search combining dense and sparse retrieval

        Args:
            query: Search query
            limit: Maximum number of results
            dense_weight: Weight for dense search (overrides default)
            sparse_weight: Weight for sparse search (overrides default)

        Returns:
            List of SearchResult objects with hybrid scoring
        """
        try:
            # Use custom weights if provided
            # Resolve effective weights for this search
            dense_weight = dense_weight if dense_weight is not None else self.default_dense_weight
            sparse_weight = sparse_weight if sparse_weight is not None else self.default_sparse_weight
            dense_weight, sparse_weight = self._normalize_weights(dense_weight, sparse_weight)

            # Track weights for stats/inspection
            self.current_dense_weight = dense_weight
            self.current_sparse_weight = sparse_weight

            # Process and expand query
            intents = self.query_processor.classify_intent(query)
            expanded_query = self.query_processor.expand_query(query, intents)

            logger.info(f"Query intents: {intents}")
            logger.debug(f"Expanded query: {expanded_query}")

            # Perform dense search (vector similarity)
            dense_results = self._dense_search(
                query=expanded_query,
                limit=limit * 2,
                desired_results=limit,
                base_threshold=self.default_dense_threshold
            )

            # Perform sparse search (TF-IDF)
            sparse_results = self._sparse_search(query, limit * 2)

            # Combine and rank results
            hybrid_results = self._combine_results(
                dense_results,
                sparse_results,
                limit,
                dense_weight,
                sparse_weight
            )

            logger.info(f"Hybrid search returned {len(hybrid_results)} results")
            return hybrid_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def _dense_search(self, query: str, limit: int, desired_results: int,
                      base_threshold: Optional[float] = None) -> Dict[str, float]:
        """Perform dense vector search"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_query_embedding(query)
            if not query_embedding:
                return {}

            thresholds = self._build_threshold_schedule(base_threshold)
            dense_scores: Dict[str, float] = {}
            desired_results = max(desired_results, 1)
            effective_threshold = thresholds[0] if thresholds else self.default_dense_threshold

            for threshold in thresholds:
                similar_chunks = self.db_manager.similarity_search(
                    query_embedding=query_embedding,
                    threshold=threshold,
                    limit=limit
                )

                for chunk in similar_chunks:
                    chunk_id = str(chunk.get('id', chunk.get('chunk_id', '')))
                    score = float(chunk.get('similarity', 0.0))
                    # Keep the best score if duplicates appear across threshold passes
                    if chunk_id not in dense_scores or score > dense_scores[chunk_id]:
                        dense_scores[chunk_id] = score

                if len(dense_scores) >= desired_results:
                    effective_threshold = threshold
                    if threshold < thresholds[0]:
                        logger.debug(
                            "Dense search relaxed threshold to %.2f to satisfy desired results",
                            threshold
                        )
                    break

            # Update the current threshold used for stats/debugging
            self.current_dense_threshold = effective_threshold

            if not dense_scores and thresholds:
                logger.warning(
                    "Dense search returned no results even after relaxing threshold to %.2f",
                    thresholds[-1]
                )

            return dense_scores

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return {}

    def _sparse_search(self, query: str, limit: int) -> Dict[str, float]:
        """Perform sparse TF-IDF search"""
        try:
            if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
                logger.warning("TF-IDF index not available")
                return {}

            # Preprocess query
            processed_query = self.query_processor.preprocess_text(query)

            # Vectorize query
            query_vector = self.tfidf_vectorizer.transform([processed_query])

            # Compute similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Get top results
            top_indices = np.argsort(similarities)[::-1][:limit]

            # Return chunk_id -> similarity score mapping
            sparse_scores = {}
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum similarity threshold
                    chunk = self.document_chunks[idx]
                    chunk_id = str(chunk['id'])
                    sparse_scores[chunk_id] = float(similarities[idx])

            return sparse_scores

        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return {}

    def _combine_results(self, dense_results: Dict[str, float],
                        sparse_results: Dict[str, float], limit: int,
                        dense_weight: float, sparse_weight: float) -> List[SearchResult]:
        """Combine dense and sparse results with hybrid scoring"""
        try:
            # Get all unique chunk IDs
            all_chunk_ids = set(dense_results.keys()) | set(sparse_results.keys())

            combined_results = []

            for chunk_id in all_chunk_ids:
                dense_score = dense_results.get(chunk_id, 0.0)
                sparse_score = sparse_results.get(chunk_id, 0.0)

                # Compute hybrid score
                hybrid_score = (dense_weight * dense_score +
                                sparse_weight * sparse_score)

                # Get chunk data
                chunk_data = self._get_chunk_data(chunk_id)
                if chunk_data:
                    result = SearchResult(
                        document_id=chunk_data['document_id'],
                        chunk_text=chunk_data['chunk_text'],
                        dense_score=dense_score,
                        sparse_score=sparse_score,
                        hybrid_score=hybrid_score,
                        chunk_index=chunk_data.get('chunk_index', 0),
                        metadata=chunk_data.get('metadata', {})
                    )
                    combined_results.append(result)

            # Sort by hybrid score and return top results
            combined_results.sort(key=lambda x: x.hybrid_score, reverse=True)
            return combined_results[:limit]

        except Exception as e:
            logger.error(f"Result combination failed: {e}")
            return []

    def _get_chunk_data(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk data by ID"""
        try:
            result = self.db_manager.client.table('document_chunks')\
                .select('*')\
                .eq('id', chunk_id)\
                .execute()

            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Failed to get chunk data for {chunk_id}: {e}")
            return None

    def update_index(self):
        """Rebuild the search index with new documents"""
        logger.info("Updating search index...")
        self._build_search_index()

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            "total_chunks": len(self.document_chunks),
            "tfidf_features": self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            "dense_weight": self.current_dense_weight,
            "sparse_weight": self.current_sparse_weight,
            "dense_threshold": self.current_dense_threshold,
            "index_status": "ready" if self.tfidf_matrix is not None else "not_ready"
        }

    def _normalize_weights(self, dense_weight: float, sparse_weight: float) -> Tuple[float, float]:
        """Ensure dense/sparse weights form a valid convex combination"""
        dense_weight = max(float(dense_weight), 0.0)
        sparse_weight = max(float(sparse_weight), 0.0)

        if dense_weight == 0 and sparse_weight == 0:
            dense_weight = self.default_dense_weight
            sparse_weight = self.default_sparse_weight

        total = dense_weight + sparse_weight
        return dense_weight / total, sparse_weight / total

    def _build_threshold_schedule(self, base_threshold: Optional[float]) -> List[float]:
        """Create a descending list of thresholds to fall back through"""
        if base_threshold is None or np.isnan(base_threshold):
            base_threshold = self.default_dense_threshold

        base_threshold = float(min(max(base_threshold, 0.05), 0.95))
        thresholds = []
        current = base_threshold

        while current >= 0.15:
            thresholds.append(round(current, 2))
            current = round(current - 0.1, 2)

        if 0.0 not in thresholds:
            thresholds.append(0.0)

        # Ensure thresholds are unique and sorted high to low
        unique_thresholds = sorted(set(thresholds), reverse=True)
        return unique_thresholds if unique_thresholds else [self.default_dense_threshold]

    # ------------------------------------------------------------------
    # Document-to-document similarity utilities
    # ------------------------------------------------------------------

    def document_similarity(self, document_a: str, document_b: str,
                             aggregation: str = "mean", top_k: int = 3) -> Optional[DocumentSimilarityResult]:
        """Compute similarity between two documents using their chunk embeddings"""
        profile_a = self._get_document_profile(document_a, aggregation)
        profile_b = self._get_document_profile(document_b, aggregation)

        if not profile_a or not profile_b:
            logger.warning(
                "Cannot compute document similarity between %s and %s due to missing embeddings",
                document_a,
                document_b
            )
            return None

        embeddings_a, chunks_a, vector_a = profile_a
        embeddings_b, chunks_b, vector_b = profile_b

        similarity = float(np.dot(vector_a, vector_b))
        similarity_matrix = self._chunk_similarity_matrix(embeddings_a, embeddings_b)

        chunk_matches = self._top_chunk_matches(
            document_a,
            chunks_a,
            document_b,
            chunks_b,
            similarity_matrix,
            top_k
        )

        stats = self._compute_similarity_stats(
            similarity_matrix,
            aggregation,
            top_k
        )

        return DocumentSimilarityResult(
            document_a=document_a,
            document_b=document_b,
            similarity=similarity,
            aggregation=aggregation,
            chunk_matches=chunk_matches,
            stats=stats
        )

    def compare_document_to_corpus(self, reference_document_id: str,
                                   candidate_document_ids: List[str],
                                   aggregation: str = "mean",
                                   top_k: int = 3) -> List[DocumentSimilarityResult]:
        """Compare a reference document against multiple candidates"""
        reference_profile = self._get_document_profile(reference_document_id, aggregation)
        if not reference_profile:
            logger.warning("Reference document %s has no embeddings", reference_document_id)
            return []

        ref_embeddings, ref_chunks, ref_vector = reference_profile
        results: List[DocumentSimilarityResult] = []

        for candidate_id in candidate_document_ids:
            if candidate_id == reference_document_id:
                continue

            candidate_profile = self._get_document_profile(candidate_id, aggregation)
            if not candidate_profile:
                logger.debug("Skipping candidate %s due to missing embeddings", candidate_id)
                continue

            cand_embeddings, cand_chunks, cand_vector = candidate_profile
            similarity = float(np.dot(ref_vector, cand_vector))
            similarity_matrix = self._chunk_similarity_matrix(ref_embeddings, cand_embeddings)

            chunk_matches = self._top_chunk_matches(
                reference_document_id,
                ref_chunks,
                candidate_id,
                cand_chunks,
                similarity_matrix,
                top_k
            )

            stats = self._compute_similarity_stats(
                similarity_matrix,
                aggregation,
                top_k
            )

            results.append(DocumentSimilarityResult(
                document_a=reference_document_id,
                document_b=candidate_id,
                similarity=similarity,
                aggregation=aggregation,
                chunk_matches=chunk_matches,
                stats=stats
            ))

        results.sort(key=lambda item: item.similarity, reverse=True)
        return results

    def _get_document_profile(self, document_id: str, aggregation: str) -> Optional[Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]]:
        """Retrieve embeddings, chunk metadata, and aggregated vector for a document"""
        embeddings, chunk_metadata = self._fetch_document_embeddings(document_id)
        if embeddings.size == 0:
            return None

        aggregated_vector = self._aggregate_document_embedding(embeddings, aggregation)
        return embeddings, chunk_metadata, aggregated_vector

    def _fetch_document_embeddings(self, document_id: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load normalized chunk embeddings and metadata for a document"""
        chunks = self.db_manager.get_document_chunks(document_id, include_embeddings=True)
        if not chunks:
            return np.empty((0, Config.EMBEDDING_DIMENSION)), []

        embeddings = []
        metadata: List[Dict[str, Any]] = []

        for chunk in chunks:
            embedding = chunk.get('embedding')
            if not embedding:
                continue

            embeddings.append(np.array(embedding, dtype=np.float32))
            metadata.append({
                "document_id": document_id,
                "chunk_id": chunk.get('id'),
                "chunk_index": chunk.get('chunk_index', 0),
                "chunk_text": chunk.get('chunk_text', ''),
                "metadata": chunk.get('metadata', {})
            })

        if not embeddings:
            return np.empty((0, Config.EMBEDDING_DIMENSION)), metadata

        embedding_matrix = np.vstack(embeddings)
        normalized = self._l2_normalize_matrix(embedding_matrix)
        return normalized, metadata

    def _aggregate_document_embedding(self, embeddings: np.ndarray, aggregation: str) -> np.ndarray:
        """Aggregate chunk embeddings into a single document vector"""
        if embeddings.size == 0:
            return np.zeros(Config.EMBEDDING_DIMENSION)

        aggregation = aggregation.lower()
        if aggregation == 'mean':
            vector = np.mean(embeddings, axis=0)
        elif aggregation == 'median':
            vector = np.median(embeddings, axis=0)
        elif aggregation == 'max':
            vector = np.max(embeddings, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        return self._l2_normalize_vector(vector)

    def _l2_normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """L2-normalize a single vector"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _l2_normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """L2-normalize each row of a matrix"""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _chunk_similarity_matrix(self, embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
        """Compute chunk-to-chunk cosine similarity matrix"""
        if embeddings_a.size == 0 or embeddings_b.size == 0:
            return np.empty((embeddings_a.shape[0], embeddings_b.shape[0]))
        return embeddings_a @ embeddings_b.T

    def _compute_similarity_stats(self, similarity_matrix: np.ndarray,
                                  aggregation: str, top_k: int) -> Dict[str, Any]:
        """Derive summary statistics from a similarity matrix"""
        if similarity_matrix.size == 0:
            return {
                "chunks_a": similarity_matrix.shape[0],
                "chunks_b": similarity_matrix.shape[1] if similarity_matrix.ndim == 2 else 0,
                "top_k": top_k,
                "aggregation": aggregation,
                "mean_chunk_similarity": 0.0,
                "max_chunk_similarity": 0.0,
                "min_chunk_similarity": 0.0,
                "std_chunk_similarity": 0.0
            }

        return {
            "chunks_a": similarity_matrix.shape[0],
            "chunks_b": similarity_matrix.shape[1],
            "top_k": top_k,
            "aggregation": aggregation,
            "mean_chunk_similarity": float(np.mean(similarity_matrix)),
            "max_chunk_similarity": float(np.max(similarity_matrix)),
            "min_chunk_similarity": float(np.min(similarity_matrix)),
            "std_chunk_similarity": float(np.std(similarity_matrix))
        }

    def _top_chunk_matches(self, document_a: str, chunks_a: List[Dict[str, Any]],
                            document_b: str, chunks_b: List[Dict[str, Any]],
                            similarity_matrix: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Identify the most similar chunk pairs across two documents"""
        if top_k <= 0 or similarity_matrix.size == 0:
            return []

        flattened = similarity_matrix.reshape(-1)
        num_pairs = flattened.size
        candidate_count = min(num_pairs, max(top_k * 5, 50))

        if num_pairs > candidate_count:
            candidate_indices = np.argpartition(flattened, -candidate_count)[-candidate_count:]
            sorted_indices = candidate_indices[np.argsort(flattened[candidate_indices])[::-1]]
        else:
            sorted_indices = np.argsort(flattened)[::-1]

        matches = []
        used_pairs = set()

        for flat_index in sorted_indices:
            row, col = divmod(int(flat_index), similarity_matrix.shape[1])
            if (row, col) in used_pairs:
                continue

            score = float(similarity_matrix[row, col])
            if np.isnan(score):
                continue

            chunk_a = chunks_a[row]
            chunk_b = chunks_b[col]

            matches.append({
                "score": score,
                "chunk_a": {
                    "document_id": document_a,
                    "chunk_id": chunk_a.get('chunk_id'),
                    "chunk_index": chunk_a.get('chunk_index'),
                    "text_preview": self._text_preview(chunk_a.get('chunk_text', '')),
                },
                "chunk_b": {
                    "document_id": document_b,
                    "chunk_id": chunk_b.get('chunk_id'),
                    "chunk_index": chunk_b.get('chunk_index'),
                    "text_preview": self._text_preview(chunk_b.get('chunk_text', '')),
                }
            })

            used_pairs.add((row, col))
            if len(matches) >= top_k:
                break

        return matches

    def _text_preview(self, text: str, max_chars: int = 160) -> str:
        """Create a short preview of chunk text for similarity reporting"""
        if not text:
            return ''

        normalized = ' '.join(text.split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip() + '...'
