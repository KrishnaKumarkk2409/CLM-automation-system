"""
Hybrid similarity search engine for CLM automation system.
Implements cosine + dot product hybrid scoring for enhanced document retrieval.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.config import Config
from src.database import DatabaseManager
from src.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for similarity search results."""

    document_id: Any
    chunk_index: int
    chunk_text: str
    similarity_cosine: float
    similarity_dot: float
    rerank_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    filename: Optional[str] = None

    def short_snippet(self, max_chars: int = 140) -> str:
        snippet = " ".join(self.chunk_text.split())
        if len(snippet) <= max_chars:
            return snippet
        return snippet[: max_chars - 3] + "..."


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Calculate dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def l2_norm(vec: Sequence[float]) -> float:
    """Calculate L2 norm (magnitude) of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    denominator = l2_norm(a) * l2_norm(b)
    if denominator == 0:
        return 0.0
    return dot_product(a, b) / denominator


def min_max_scale(values: Sequence[float]) -> List[float]:
    """Scale values to [0, 1] range using min-max normalization."""
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_max, v_min):
        # fallback to neutral weight if all dot scores identical
        return [0.5 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


class HybridSimilarityEngine:
    """
    Hybrid similarity search engine combining cosine and dot product scoring.
    Provides enhanced document retrieval with configurable scoring weights.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_manager: EmbeddingManager,
        candidate_pool: int = 50,
        cosine_weight: float = 0.7,
    ):
        """
        Initialize the hybrid similarity engine.

        Args:
            db_manager: Database manager instance
            embedding_manager: Embedding manager instance
            candidate_pool: Number of candidates to retrieve before reranking (0 = scan all)
            cosine_weight: Weight for cosine similarity in hybrid score (0-1)
        """
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.candidate_pool = candidate_pool
        self.cosine_weight = cosine_weight

        logger.info(
            f"Hybrid similarity engine initialized with candidate_pool={candidate_pool}, "
            f"cosine_weight={cosine_weight}"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.6,
        include_metadata: bool = True,
    ) -> Tuple[List[SearchResult], List[float]]:
        """
        Perform hybrid similarity search.

        Args:
            query: Natural language query
            top_k: Number of top results to return after reranking
            similarity_threshold: Minimum cosine similarity threshold
            include_metadata: Whether to include full metadata in results

        Returns:
            Tuple of (search results, query embedding)
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_query_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return [], []

            # Fetch candidate chunks
            records = self._fetch_candidates(query_embedding)

            if not records:
                logger.info("No candidate chunks found")
                return [], query_embedding

            # Calculate similarity scores
            dot_scores: List[float] = []
            results: List[SearchResult] = []

            for record in records:
                embedding = record.get("embedding") or record.get("embedding_vector")
                if not embedding:
                    continue

                dot_score = dot_product(query_embedding, embedding)
                cosine_score = cosine_similarity(query_embedding, embedding)
                dot_scores.append(dot_score)

                results.append(
                    SearchResult(
                        document_id=record.get("document_id"),
                        chunk_index=record.get("chunk_index", 0),
                        chunk_text=record.get("chunk_text", ""),
                        similarity_cosine=cosine_score,
                        similarity_dot=dot_score,
                        rerank_score=0.0,  # Will be calculated next
                        metadata={"raw_record": record} if include_metadata else {},
                    )
                )

            # Scale dot product scores and compute hybrid rerank scores
            scaled_dots = min_max_scale(dot_scores)
            for result, scaled_dot in zip(results, scaled_dots):
                result.rerank_score = (
                    self.cosine_weight * result.similarity_cosine +
                    (1 - self.cosine_weight) * scaled_dot
                )

            # Filter by similarity threshold and sort by rerank score
            filtered = [
                res for res in results
                if res.similarity_cosine >= similarity_threshold
            ]
            filtered.sort(key=lambda res: res.rerank_score, reverse=True)
            top_results = filtered[:top_k]

            # Hydrate document information
            self._hydrate_documents(top_results)

            logger.info(
                f"Hybrid search completed: {len(records)} candidates -> "
                f"{len(filtered)} above threshold -> {len(top_results)} returned"
            )

            return top_results, query_embedding

        except Exception as e:
            logger.error(f"Hybrid similarity search failed: {e}", exc_info=True)
            return [], []

    def _fetch_candidates(self, query_embedding: Sequence[float]) -> List[Dict[str, Any]]:
        """Fetch candidate chunks from database."""
        if self.candidate_pool <= 0:
            return self._fetch_all_chunks()

        # Try vector search via Supabase RPC if available
        try:
            response = self.db_manager.client.rpc(
                "match_document_chunks",
                {
                    "query_embedding": list(query_embedding),
                    "match_threshold": 0,
                    "match_count": self.candidate_pool,
                },
            ).execute()
            data = getattr(response, "data", None)
            if data:
                logger.debug(f"Retrieved {len(data)} candidates via RPC vector search")
                return data
        except Exception as exc:
            logger.debug(f"RPC vector search failed, falling back to regular fetch: {exc}")

        return self._fetch_all_chunks(limit=self.candidate_pool)

    def _fetch_all_chunks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch chunks directly from database table."""
        try:
            query = self.db_manager.client.table("document_chunks").select(
                "id, document_id, chunk_index, chunk_text, embedding"
            )
            if limit:
                query = query.limit(limit)
            response = query.execute()
            data = response.data or []
            logger.debug(f"Fetched {len(data)} chunks from database")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch chunks: {e}")
            return []

    def _hydrate_documents(self, results: Iterable[SearchResult]) -> None:
        """Add document metadata to search results."""
        doc_ids = {res.document_id for res in results if res.document_id is not None}
        if not doc_ids:
            return

        try:
            response = (
                self.db_manager.client.table("documents")
                .select("id, filename, content, metadata")
                .in_("id", list(doc_ids))
                .execute()
            )
            if not response.data:
                return

            doc_map = {row["id"]: row for row in response.data}
            for result in results:
                document = doc_map.get(result.document_id)
                if not document:
                    continue
                result.filename = document.get("filename") or str(document.get("id"))
                result.metadata["document"] = document

        except Exception as e:
            logger.error(f"Failed to hydrate document metadata: {e}")

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search and return enriched results.

        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Minimum similarity threshold

        Returns:
            List of enriched search results with document metadata
        """
        results, _ = self.search(
            query=query,
            top_k=limit,
            similarity_threshold=threshold,
            include_metadata=True
        )

        # Convert to dictionary format for backward compatibility
        enriched_results = []
        for result in results:
            enriched_results.append({
                "chunk_text": result.chunk_text,
                "similarity": result.similarity_cosine,
                "rerank_score": result.rerank_score,
                "document_id": result.document_id,
                "filename": result.filename or "Unknown",
                "chunk_index": result.chunk_index,
            })

        return enriched_results
