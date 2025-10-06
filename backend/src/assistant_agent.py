"""Conversational agent orchestrating contract assistance.

Routes simple interactions without invoking the RAG pipeline and calls
into the enhanced retrieval stack only when document context is needed.
"""

import logging
import random
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AssistantAgent:
    """Coordinate greetings, help, analytics tasks, and RAG lookups."""

    GREETING_KEYWORDS = {
        "hi",
        "hello",
        "hey",
        "hola",
        "hoi",
        "howdy",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "hlo",
        "hoi",
    }

    HELP_KEYWORDS = {
        "help",
        "capabilities",
        "what can you do",
        "how do i",
        "how to",
        "assist",
        "support",
    }

    AGENT_KEYWORDS = {
        "expiring",
        "renew",
        "renewal",
        "conflict",
        "summary",
        "report",
        "monitor",
        "status",
        "analytics",
        "dashboard",
    }

    SMALLTALK_RESPONSES = [
        "I'm Padaku.ai how can i help-you?",
    ]

    SMALLTALK_KEYWORDS = {
        "ok",
        "okay",
        "cool",
        "thanks",
        "thank",
        "sure",
        "awesome",
        "great",
        "nice",
        "yup",
        "yeah",
        "haha",
        "lol",
        "alright",
        "hm",
        "hmm",
        "hlo",
        "hey",
        "hi",
        "hello",
    }

    RETRIEVAL_KEYWORDS = {
        "find",
        "search",
        "show",
        "locate",
        "lookup",
        "list",
        "tell",
        "about",
        "give",
        "show me",
        "what",
        "who",
        "where",
        "document",
        "contract",
        "clause",
    }

    UUID_PATTERN = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    )

    def __init__(self, rag_pipeline, contract_agent, db_manager):
        self.rag_pipeline = rag_pipeline
        self.contract_agent = contract_agent
        self.db_manager = db_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, any]:
        """Process an incoming message and decide which tool to call."""
        cleaned = (message or "").strip()
        if not cleaned:
            return {
                "answer": "Please share a bit more detail so I know how to help.",
                "sources": [],
                "metadata": {"intent": "empty"},
            }

        classification = self._classify_message(cleaned)
        intent = classification["intent"]

        if intent == "greeting":
            return self._greeting_response()

        if intent == "help":
            return self._help_response()

        if intent == "agent_task":
            return self._handle_agent_task(cleaned)

        if intent == "document_similarity":
            return self._handle_document_similarity(
                classification["doc_ids"], classification.get("aggregation")
            )

        if intent == "document_similarity_single":
            return self._handle_similarity_to_corpus(
                classification["doc_ids"][0],
                classification.get("aggregation"),
                classification.get("limit", 5),
            )

        if intent == "smalltalk":
            return self._smalltalk_response()

        # Default: delegate to RAG pipeline
        logger.debug("AssistantAgent delegating message to RAG pipeline", extra={"intent": intent})
        response = self.rag_pipeline.query(cleaned)
        response.setdefault("metadata", {})["intent"] = intent
        return response

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _greeting_response(self) -> Dict[str, any]:
        answer = (
            "Hello! I'm Padaku.ai RAG assistant. "
            "Ask me anything about NCERT Books"
        )
        return {"answer": answer, "sources": [], "metadata": {"intent": "greeting"}}

    def _help_response(self) -> Dict[str, any]:
        answer = (
            "Here's what I can help with:\n"
            "• Search & retrieval – e.g. 'Show payment terms for the ACME contract'\n"
            "• Monitoring – e.g. 'Contracts expiring in 60 days'\n"
            "• Comparisons – e.g. 'Compare document <id1> with <id2>'\n"
            "• Similarity discovery – e.g. 'Find documents similar to <id>'"
        )
        return {"answer": answer, "sources": [], "metadata": {"intent": "help"}}

    def _smalltalk_response(self) -> Dict[str, any]:
        answer = random.choice(self.SMALLTALK_RESPONSES)
        return {"answer": answer, "sources": [], "metadata": {"intent": "smalltalk"}}

    def _handle_agent_task(self, message: str) -> Dict[str, any]:
        try:
            reply = self.contract_agent.query_agent(message)
            return {
                "answer": f"Contract agent response: {reply}",
                "sources": [],
                "metadata": {"intent": "agent_task"},
            }
        except Exception as exc:
            logger.error("Contract agent failed", exc_info=exc)
            return {
                "answer": "I hit an error while handling that request. Please try again shortly.",
                "sources": [],
                "metadata": {"intent": "agent_task", "error": str(exc)},
            }

    def _handle_document_similarity(
        self, doc_ids: Tuple[str, str], aggregation: Optional[str]
    ) -> Dict[str, any]:
        doc_a, doc_b = doc_ids
        aggregation = aggregation or "mean"
        payload = self.rag_pipeline.document_similarity(
            document_a=doc_a,
            document_b=doc_b,
            aggregation=aggregation,
            top_k=3,
        )

        if not payload:
            return {
                "answer": "I could not compute similarity for those document IDs. Please make sure they exist.",
                "sources": [],
                "metadata": {"intent": "document_similarity", "status": "not_found"},
            }

        answer = self._format_similarity_result(payload)
        sources = self._format_similarity_sources(payload)
        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "intent": "document_similarity",
                "aggregation": payload.get("aggregation"),
                "similarity": payload.get("similarity"),
            },
        }

    def _handle_similarity_to_corpus(
        self, doc_id: str, aggregation: Optional[str], limit: int
    ) -> Dict[str, any]:
        aggregation = aggregation or "mean"
        limit = max(1, min(limit or 5, 10))
        comparisons = self.rag_pipeline.compare_document_to_corpus(
            reference_document_id=doc_id,
            limit=limit,
            aggregation=aggregation,
            top_k=3,
        )

        if not comparisons:
            return {
                "answer": "I could not find similar documents. Ensure the reference document has embeddings.",
                "sources": [],
                "metadata": {"intent": "document_similarity_single", "status": "not_found"},
            }

        lines = [
            f"Top {len(comparisons)} documents similar to {doc_id} (aggregation: {aggregation}):"
        ]
        for idx, item in enumerate(comparisons, start=1):
            doc_info = item.get("documents", {}).get("document_b", {})
            score = item.get("similarity", 0.0)
            lines.append(
                f"{idx}. {doc_info.get('filename', 'Unnamed document')} — {doc_info.get('id')} (score: {score:.3f})"
            )

        sources = self._format_similarity_sources(comparisons[0]) if comparisons else []
        return {
            "answer": "\n".join(lines),
            "sources": sources,
            "metadata": {
                "intent": "document_similarity_single",
                "aggregation": aggregation,
                "count": len(comparisons),
            },
        }

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _classify_message(self, message: str) -> Dict[str, any]:
        lower = message.lower()

        similarity = self._detect_similarity_request(lower)
        if similarity:
            return similarity

        if self._is_greeting(lower):
            return {"intent": "greeting"}

        if any(keyword in lower for keyword in self.HELP_KEYWORDS):
            return {"intent": "help"}

        if any(keyword in lower for keyword in self.AGENT_KEYWORDS):
            return {"intent": "agent_task"}

        if self._looks_like_smalltalk(lower):
            return {"intent": "smalltalk"}

        if self._looks_like_retrieval(lower):
            return {"intent": "document_query"}

        return {"intent": "document_query"}

    def _detect_similarity_request(self, text: str) -> Optional[Dict[str, any]]:
        doc_ids = self.UUID_PATTERN.findall(text)
        if len(doc_ids) >= 2 and "compare" in text:
            aggregation = self._extract_aggregation_keyword(text)
            return {
                "intent": "document_similarity",
                "doc_ids": (doc_ids[0], doc_ids[1]),
                "aggregation": aggregation,
            }

        if len(doc_ids) == 1 and any(keyword in text for keyword in {"find similar", "similar to", "like", "match"}):
            aggregation = self._extract_aggregation_keyword(text)
            limit = 5
            match = re.search(r"top\s+(\d+)", text)
            if match:
                try:
                    limit = int(match.group(1))
                except ValueError:
                    pass
            return {
                "intent": "document_similarity_single",
                "doc_ids": (doc_ids[0],),
                "aggregation": aggregation,
                "limit": limit,
            }

        return None

    def _extract_aggregation_keyword(self, text: str) -> Optional[str]:
        for candidate in {"mean", "median", "max"}:
            if candidate in text:
                return candidate
        return None

    def _is_greeting(self, text: str) -> bool:
        tokens = self._tokenize(text)
        for token in tokens:
            if token in self.GREETING_KEYWORDS:
                return True
            if len(token) < 3:
                continue
            for keyword in self.GREETING_KEYWORDS:
                if len(keyword) < 3:
                    continue
                if SequenceMatcher(None, token, keyword).ratio() >= 0.85:
                    return True
        return False

    def _looks_like_smalltalk(self, text: str) -> bool:
        tokens = self._tokenize(text)
        if not tokens or len(tokens) > 3:
            return False

        if all(token in self.SMALLTALK_KEYWORDS for token in tokens):
            return True

        if len(tokens) <= 2 and not self._looks_like_retrieval(text):
            return True

        return False

    def _looks_like_retrieval(self, text: str) -> bool:
        return any(keyword in text for keyword in self.RETRIEVAL_KEYWORDS)

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_similarity_result(self, payload: Dict[str, any]) -> str:
        documents = payload.get("documents", {})
        doc_a = documents.get("document_a", {})
        doc_b = documents.get("document_b", {})
        similarity = payload.get("similarity", 0.0)
        aggregation = payload.get("aggregation")

        lines = [
            f"Document similarity (aggregation: {aggregation}):",
            f"- {doc_a.get('filename', 'Unknown')} — {doc_a.get('id')}",
            f"- {doc_b.get('filename', 'Unknown')} — {doc_b.get('id')}",
            f"Similarity score: {similarity:.3f}",
        ]

        matches = payload.get("chunk_matches", [])
        if not matches:
            lines.append("No overlapping sections detected.")
            return "\n".join(lines)

        lines.append("Top overlapping sections:")
        for idx, match in enumerate(matches, start=1):
            chunk_a = match.get("chunk_a", {})
            chunk_b = match.get("chunk_b", {})
            lines.append(
                f"{idx}. Score {match.get('score', 0.0):.3f}\n"
                f"   A#{chunk_a.get('chunk_index')} » {chunk_a.get('text_preview', '')}\n"
                f"   B#{chunk_b.get('chunk_index')} » {chunk_b.get('text_preview', '')}"
            )

        stats = payload.get("stats", {})
        if stats:
            lines.append(
                "Stats: "
                f"mean={stats.get('mean_chunk_similarity', 0.0):.3f}, "
                f"max={stats.get('max_chunk_similarity', 0.0):.3f}, "
                f"min={stats.get('min_chunk_similarity', 0.0):.3f}"
            )

        return "\n".join(lines)

    def _format_similarity_sources(self, payload: Dict[str, any]) -> List[Dict[str, any]]:
        matches = payload.get("chunk_matches", [])
        return [
            {
                "id": idx,
                "score": match.get("score"),
                "chunk_a": match.get("chunk_a"),
                "chunk_b": match.get("chunk_b"),
            }
            for idx, match in enumerate(matches, start=1)
        ]
