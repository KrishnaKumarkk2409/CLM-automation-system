"""
Advanced Re-ranking Module for CLM automation system.
Implements cross-encoder re-ranking and multi-factor relevance scoring.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

@dataclass
class RerankingFeatures:
    """Features used for re-ranking"""
    semantic_similarity: float
    keyword_match_score: float
    document_relevance: float
    temporal_relevance: float
    structural_importance: float
    entity_match_score: float
    legal_term_relevance: float
    user_feedback_score: float

@dataclass
class RankedResult:
    """Enhanced search result with detailed scoring"""
    document_id: str
    chunk_text: str
    original_score: float
    reranked_score: float
    features: RerankingFeatures
    chunk_metadata: Dict[str, Any]
    ranking_explanation: str

class LegalTermAnalyzer:
    """Analyzes legal terms and their importance"""

    def __init__(self):
        # Legal term categories with importance weights
        self.legal_term_categories = {
            'obligations': {
                'terms': ['shall', 'must', 'required', 'obligation', 'duty', 'covenant'],
                'weight': 0.9
            },
            'rights': {
                'terms': ['right', 'entitled', 'privilege', 'authority', 'power'],
                'weight': 0.8
            },
            'prohibitions': {
                'terms': ['shall not', 'must not', 'prohibited', 'forbidden', 'restricted'],
                'weight': 0.9
            },
            'financial': {
                'terms': ['payment', 'fee', 'cost', 'expense', 'compensation', 'penalty'],
                'weight': 0.8
            },
            'temporal': {
                'terms': ['term', 'duration', 'expiry', 'deadline', 'renewal', 'effective'],
                'weight': 0.7
            },
            'termination': {
                'terms': ['terminate', 'end', 'cancel', 'breach', 'default', 'violation'],
                'weight': 0.9
            },
            'liability': {
                'terms': ['liable', 'responsible', 'indemnify', 'damages', 'loss'],
                'weight': 0.8
            },
            'intellectual_property': {
                'terms': ['copyright', 'trademark', 'patent', 'trade secret', 'proprietary'],
                'weight': 0.7
            }
        }

        logger.info("Legal term analyzer initialized")

    def analyze_legal_terms(self, query: str, text: str) -> float:
        """Analyze relevance of legal terms in text relative to query"""
        query_lower = query.lower()
        text_lower = text.lower()

        total_score = 0.0
        matches_found = 0

        for category, data in self.legal_term_categories.items():
            category_score = 0.0
            category_matches = 0

            for term in data['terms']:
                # Check if term appears in both query and text
                if term in query_lower and term in text_lower:
                    # Count occurrences in text
                    text_count = text_lower.count(term)
                    query_count = query_lower.count(term)

                    # Score based on frequency and category weight
                    term_score = min(text_count / 5.0, 1.0) * data['weight']
                    category_score += term_score
                    category_matches += 1

            if category_matches > 0:
                total_score += category_score / category_matches
                matches_found += 1

        return total_score / max(matches_found, 1)

class EntityMatcher:
    """Matches entities between query and documents"""

    def __init__(self):
        # Entity patterns
        self.entity_patterns = {
            'company': r'\b(?:[A-Z][a-z]*\s*)+(?:Corp|Corporation|Inc|LLC|Ltd|Company|Co\.)\b',
            'person': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            'money': r'\$[\d,]+(?:\.\d{2})?|\b\d+\s*(?:dollars?|USD)\b',
            'percentage': r'\d+(?:\.\d+)?%',
            'contract_ref': r'(?:Agreement|Contract|Amendment)\s+(?:No\.?\s*)?\w+',
            'section_ref': r'Section\s+\d+(?:\.\d+)*|Article\s+\d+'
        }

        logger.info("Entity matcher initialized")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {}

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches))  # Remove duplicates

        return entities

    def calculate_entity_overlap(self, query: str, text: str) -> float:
        """Calculate entity overlap between query and text"""
        query_entities = self.extract_entities(query)
        text_entities = self.extract_entities(text)

        total_overlap = 0.0
        entity_types_checked = 0

        for entity_type in query_entities:
            if entity_type in text_entities:
                query_set = set(e.lower() for e in query_entities[entity_type])
                text_set = set(e.lower() for e in text_entities[entity_type])

                if query_set and text_set:
                    overlap = len(query_set.intersection(text_set)) / len(query_set.union(text_set))
                    total_overlap += overlap
                    entity_types_checked += 1

        return total_overlap / max(entity_types_checked, 1)

class TemporalRelevanceCalculator:
    """Calculates temporal relevance for date-sensitive queries"""

    def __init__(self):
        self.temporal_keywords = [
            'expiring', 'expire', 'renewal', 'due', 'upcoming',
            'recent', 'current', 'active', 'past', 'future'
        ]

    def calculate_temporal_relevance(self, query: str, text: str,
                                   document_metadata: Dict[str, Any]) -> float:
        """Calculate temporal relevance score"""
        query_lower = query.lower()

        # Check if query has temporal intent
        has_temporal_intent = any(keyword in query_lower for keyword in self.temporal_keywords)

        if not has_temporal_intent:
            return 0.5  # Neutral score for non-temporal queries

        # Extract dates from text
        dates = self._extract_dates(text)
        current_date = datetime.now()

        if not dates:
            return 0.3  # Lower score if no dates found in temporal query

        # Calculate relevance based on date proximity and query intent
        relevance_scores = []

        for date_obj in dates:
            days_diff = (date_obj - current_date).days

            if 'expiring' in query_lower or 'expire' in query_lower:
                # Future dates are more relevant for expiring queries
                if 0 <= days_diff <= 90:  # Within 3 months
                    score = 1.0 - (days_diff / 90) * 0.5
                elif days_diff > 90:
                    score = 0.3
                else:
                    score = 0.1  # Past dates
            elif 'renewal' in query_lower:
                # Dates close to renewal periods
                if -30 <= days_diff <= 30:  # Within 1 month
                    score = 1.0 - abs(days_diff) / 30 * 0.3
                else:
                    score = 0.4
            elif 'recent' in query_lower or 'current' in query_lower:
                # Recent dates are more relevant
                if days_diff <= 0 and abs(days_diff) <= 30:
                    score = 1.0 - abs(days_diff) / 30 * 0.4
                else:
                    score = 0.3
            else:
                # General temporal relevance
                score = max(0.1, 1.0 - abs(days_diff) / 365)

            relevance_scores.append(score)

        return max(relevance_scores) if relevance_scores else 0.3

    def _extract_dates(self, text: str) -> List[datetime]:
        """Extract datetime objects from text"""
        dates = []

        # Pattern for MM/DD/YYYY, MM-DD-YYYY
        date_pattern1 = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'
        matches1 = re.findall(date_pattern1, text)

        for match in matches1:
            try:
                month, day, year = map(int, match)
                if 1 <= month <= 12 and 1 <= day <= 31:
                    dates.append(datetime(year, month, day))
            except ValueError:
                continue

        # Pattern for "Month DD, YYYY"
        month_names = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }

        date_pattern2 = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b'
        matches2 = re.findall(date_pattern2, text, re.IGNORECASE)

        for match in matches2:
            try:
                month_name, day, year = match
                month = month_names[month_name.lower()]
                day, year = int(day), int(year)
                if 1 <= day <= 31:
                    dates.append(datetime(year, month, day))
            except (ValueError, KeyError):
                continue

        return dates

class AdvancedReranker:
    """Advanced re-ranking system with multiple relevance factors"""

    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.legal_analyzer = LegalTermAnalyzer()
        self.entity_matcher = EntityMatcher()
        self.temporal_calculator = TemporalRelevanceCalculator()

        # Feature weights (can be tuned)
        self.feature_weights = {
            'semantic_similarity': 0.25,
            'keyword_match': 0.20,
            'document_relevance': 0.15,
            'temporal_relevance': 0.10,
            'structural_importance': 0.10,
            'entity_match': 0.10,
            'legal_term_relevance': 0.10,
            'user_feedback': 0.0  # Can be enabled with user feedback data
        }

        logger.info("Advanced reranker initialized")

    def rerank_results(self, query: str, search_results: List[Dict[str, Any]],
                      user_context: Dict[str, Any] = None) -> List[RankedResult]:
        """
        Re-rank search results using multiple relevance factors

        Args:
            query: Original search query
            search_results: List of search results to re-rank
            user_context: Optional user context for personalization

        Returns:
            List of RankedResult objects sorted by relevance
        """
        try:
            reranked_results = []

            for result in search_results:
                # Extract features
                features = self._extract_features(query, result, user_context)

                # Calculate final score
                final_score = self._calculate_weighted_score(features)

                # Generate explanation
                explanation = self._generate_explanation(features)

                reranked_result = RankedResult(
                    document_id=result.get('document_id', ''),
                    chunk_text=result.get('chunk_text', ''),
                    original_score=result.get('similarity', 0.0),
                    reranked_score=final_score,
                    features=features,
                    chunk_metadata=result.get('metadata', {}),
                    ranking_explanation=explanation
                )

                reranked_results.append(reranked_result)

            # Sort by reranked score
            reranked_results.sort(key=lambda x: x.reranked_score, reverse=True)

            logger.info(f"Re-ranked {len(reranked_results)} results")
            return reranked_results

        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            # Fallback to original order
            return self._create_fallback_results(search_results)

    def _extract_features(self, query: str, result: Dict[str, Any],
                         user_context: Dict[str, Any] = None) -> RerankingFeatures:
        """Extract all relevance features for a result"""
        text = result.get('chunk_text', '')
        metadata = result.get('metadata', {})

        # Semantic similarity (from original search)
        semantic_similarity = result.get('similarity', 0.0)

        # Keyword match score
        keyword_match_score = self._calculate_keyword_match(query, text)

        # Document relevance
        document_relevance = self._calculate_document_relevance(result, user_context)

        # Temporal relevance
        temporal_relevance = self.temporal_calculator.calculate_temporal_relevance(
            query, text, metadata
        )

        # Structural importance
        structural_importance = self._calculate_structural_importance(result)

        # Entity match score
        entity_match_score = self.entity_matcher.calculate_entity_overlap(query, text)

        # Legal term relevance
        legal_term_relevance = self.legal_analyzer.analyze_legal_terms(query, text)

        # User feedback score (placeholder for future implementation)
        user_feedback_score = self._get_user_feedback_score(result, user_context)

        return RerankingFeatures(
            semantic_similarity=semantic_similarity,
            keyword_match_score=keyword_match_score,
            document_relevance=document_relevance,
            temporal_relevance=temporal_relevance,
            structural_importance=structural_importance,
            entity_match_score=entity_match_score,
            legal_term_relevance=legal_term_relevance,
            user_feedback_score=user_feedback_score
        )

    def _calculate_keyword_match(self, query: str, text: str) -> float:
        """Calculate keyword match score with position and frequency weighting"""
        query_words = set(query.lower().split())
        text_lower = text.lower()
        text_words = text_lower.split()

        if not query_words or not text_words:
            return 0.0

        # Exact matches
        exact_matches = sum(1 for word in query_words if word in text_lower)
        exact_score = exact_matches / len(query_words)

        # Position bonus (keywords appearing early get higher scores)
        position_bonus = 0.0
        for word in query_words:
            if word in text_lower:
                # Find first occurrence position
                position = text_lower.find(word)
                if position != -1:
                    # Normalize position (earlier = higher bonus)
                    position_score = max(0, 1 - position / len(text_lower))
                    position_bonus += position_score

        position_bonus /= len(query_words)

        # Frequency bonus
        frequency_bonus = 0.0
        for word in query_words:
            count = text_lower.count(word)
            frequency_bonus += min(count / 5.0, 1.0)  # Cap at 5 occurrences

        frequency_bonus /= len(query_words)

        # Combined score
        return (exact_score * 0.5 + position_bonus * 0.3 + frequency_bonus * 0.2)

    def _calculate_document_relevance(self, result: Dict[str, Any],
                                    user_context: Dict[str, Any] = None) -> float:
        """Calculate document-level relevance"""
        score = 0.5  # Base score

        # Document type relevance
        metadata = result.get('metadata', {})
        chunk_type = metadata.get('chunk_type', '')

        type_scores = {
            'section': 0.8,
            'clause': 0.7,
            'signature': 0.9,
            'paragraph': 0.5,
            'list_item': 0.4
        }

        score += type_scores.get(chunk_type, 0.0) * 0.3

        # Importance score from chunk metadata
        importance = metadata.get('importance_score', 0.5)
        score += importance * 0.4

        # Document freshness (if available)
        if 'document_date' in metadata:
            # Newer documents get slight bonus
            try:
                doc_date = datetime.fromisoformat(metadata['document_date'])
                days_old = (datetime.now() - doc_date).days
                freshness_score = max(0, 1 - days_old / 365)  # Linear decay over 1 year
                score += freshness_score * 0.3
            except:
                pass

        return min(score, 1.0)

    def _calculate_structural_importance(self, result: Dict[str, Any]) -> float:
        """Calculate importance based on document structure"""
        metadata = result.get('metadata', {})

        # Base importance from metadata
        base_importance = metadata.get('importance_score', 0.5)

        # Section title bonus
        section_title = result.get('section_title', '')
        title_bonus = 0.0

        if section_title:
            important_sections = [
                'termination', 'liability', 'payment', 'intellectual property',
                'confidentiality', 'governing law', 'dispute resolution'
            ]

            section_lower = section_title.lower()
            for important_section in important_sections:
                if important_section in section_lower:
                    title_bonus += 0.2

        # Length penalty for very long chunks (may be less focused)
        text_length = len(result.get('chunk_text', ''))
        length_penalty = 0.0
        if text_length > 2000:
            length_penalty = (text_length - 2000) / 10000  # Gradual penalty

        final_score = base_importance + title_bonus - length_penalty
        return max(0.0, min(final_score, 1.0))

    def _get_user_feedback_score(self, result: Dict[str, Any],
                                user_context: Dict[str, Any] = None) -> float:
        """Get user feedback score (placeholder for future implementation)"""
        # This would integrate with user feedback data
        # For now, return neutral score
        return 0.5

    def _calculate_weighted_score(self, features: RerankingFeatures) -> float:
        """Calculate final weighted score from all features"""
        score = 0.0

        score += features.semantic_similarity * self.feature_weights['semantic_similarity']
        score += features.keyword_match_score * self.feature_weights['keyword_match']
        score += features.document_relevance * self.feature_weights['document_relevance']
        score += features.temporal_relevance * self.feature_weights['temporal_relevance']
        score += features.structural_importance * self.feature_weights['structural_importance']
        score += features.entity_match_score * self.feature_weights['entity_match']
        score += features.legal_term_relevance * self.feature_weights['legal_term_relevance']
        score += features.user_feedback_score * self.feature_weights['user_feedback']

        return min(score, 1.0)

    def _generate_explanation(self, features: RerankingFeatures) -> str:
        """Generate human-readable explanation for ranking"""
        explanations = []

        if features.semantic_similarity > 0.8:
            explanations.append("High semantic similarity to query")
        elif features.semantic_similarity > 0.6:
            explanations.append("Good semantic match")

        if features.keyword_match_score > 0.7:
            explanations.append("Strong keyword matches")

        if features.entity_match_score > 0.5:
            explanations.append("Matching entities found")

        if features.legal_term_relevance > 0.6:
            explanations.append("Relevant legal terms")

        if features.temporal_relevance > 0.7:
            explanations.append("Temporally relevant")

        if features.structural_importance > 0.8:
            explanations.append("Important document section")

        return "; ".join(explanations) if explanations else "Standard relevance match"

    def _create_fallback_results(self, search_results: List[Dict[str, Any]]) -> List[RankedResult]:
        """Create fallback results when re-ranking fails"""
        fallback_results = []

        for i, result in enumerate(search_results):
            fallback_result = RankedResult(
                document_id=result.get('document_id', ''),
                chunk_text=result.get('chunk_text', ''),
                original_score=result.get('similarity', 0.0),
                reranked_score=result.get('similarity', 0.0),
                features=RerankingFeatures(
                    semantic_similarity=result.get('similarity', 0.0),
                    keyword_match_score=0.5,
                    document_relevance=0.5,
                    temporal_relevance=0.5,
                    structural_importance=0.5,
                    entity_match_score=0.5,
                    legal_term_relevance=0.5,
                    user_feedback_score=0.5
                ),
                chunk_metadata=result.get('metadata', {}),
                ranking_explanation="Fallback ranking (original order)"
            )
            fallback_results.append(fallback_result)

        return fallback_results

    def update_feature_weights(self, new_weights: Dict[str, float]):
        """Update feature weights for tuning"""
        for feature, weight in new_weights.items():
            if feature in self.feature_weights:
                self.feature_weights[feature] = weight

        # Normalize weights to sum to 1.0
        total_weight = sum(self.feature_weights.values())
        if total_weight > 0:
            for feature in self.feature_weights:
                self.feature_weights[feature] /= total_weight

        logger.info(f"Updated feature weights: {self.feature_weights}")

    def get_ranking_stats(self, results: List[RankedResult]) -> Dict[str, Any]:
        """Get statistics about ranking performance"""
        if not results:
            return {}

        # Average scores by feature
        avg_features = {}
        for feature_name in ['semantic_similarity', 'keyword_match_score', 'document_relevance',
                           'temporal_relevance', 'structural_importance', 'entity_match_score',
                           'legal_term_relevance', 'user_feedback_score']:
            scores = [getattr(r.features, feature_name) for r in results]
            avg_features[f'avg_{feature_name}'] = sum(scores) / len(scores)

        # Score distribution
        reranked_scores = [r.reranked_score for r in results]
        original_scores = [r.original_score for r in results]

        return {
            'total_results': len(results),
            'avg_reranked_score': sum(reranked_scores) / len(reranked_scores),
            'avg_original_score': sum(original_scores) / len(original_scores),
            'score_improvement': (sum(reranked_scores) - sum(original_scores)) / len(results),
            **avg_features
        }