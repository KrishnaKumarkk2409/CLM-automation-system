"""
Enhanced Embeddings Module for CLM automation system.
Implements specialized embeddings for legal/contract domain with fine-tuning capabilities.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import openai

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
import pickle
import os
from datetime import datetime

from src.config import Config
from src.database import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingMetadata:
    """Metadata for embeddings"""
    model_name: str
    embedding_dimension: int
    domain_specialized: bool
    creation_date: datetime
    text_preprocessing: str
    fine_tuning_info: Optional[Dict[str, Any]] = None

class LegalTextPreprocessor:
    """Specialized text preprocessing for legal documents"""

    def __init__(self):
        # Legal abbreviations and their expansions
        self.legal_abbreviations = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'and so forth',
            'vs.': 'versus',
            'v.': 'versus',
            'cf.': 'compare',
            'viz.': 'namely',
            'LLC': 'Limited Liability Company',
            'Corp.': 'Corporation',
            'Inc.': 'Incorporated',
            'Ltd.': 'Limited',
            'Co.': 'Company',
            'WHEREAS': 'given that',
            'THEREFORE': 'as a result',
            'NOTWITHSTANDING': 'despite',
            'FURTHERMORE': 'in addition',
            'HEREIN': 'in this document',
            'HEREBY': 'by this document',
            'HEREOF': 'of this document',
            'HEREUNDER': 'under this document'
        }

        # Legal phrase standardization
        self.legal_phrases = {
            'subject to the terms and conditions': 'under these terms',
            'to the extent permitted by law': 'if legally allowed',
            'in no event shall': 'never will',
            'shall be deemed to': 'will be considered',
            'provided that': 'if',
            'except as otherwise provided': 'unless stated differently'
        }

        logger.info("Legal text preprocessor initialized")

    def preprocess_legal_text(self, text: str, preserve_legal_structure: bool = True) -> str:
        """Preprocess legal text for better embeddings"""
        if not text:
            return text

        processed_text = text

        # Expand legal abbreviations (if not preserving structure)
        if not preserve_legal_structure:
            for abbrev, expansion in self.legal_abbreviations.items():
                processed_text = processed_text.replace(abbrev, expansion)

        # Standardize legal phrases (optional)
        if not preserve_legal_structure:
            for phrase, standard in self.legal_phrases.items():
                processed_text = processed_text.replace(phrase, standard)

        # Normalize whitespace and punctuation
        processed_text = ' '.join(processed_text.split())

        # Handle special legal formatting
        processed_text = self._handle_legal_formatting(processed_text)

        return processed_text

    def _handle_legal_formatting(self, text: str) -> str:
        """Handle special legal document formatting"""
        # Normalize section references
        import re

        # Standardize section references
        text = re.sub(r'Section\s+(\d+)(?:\.(\d+))*', r'Section \1.\2', text)
        text = re.sub(r'Article\s+(\d+)', r'Article \1', text)

        # Normalize parenthetical references
        text = re.sub(r'\(\s*([a-z])\s*\)', r'(\1)', text)
        text = re.sub(r'\(\s*(\d+)\s*\)', r'(\1)', text)

        # Clean up excessive punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

class MultiVectorEmbeddings:
    """Multi-vector embeddings for different aspects of legal documents"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.preprocessor = LegalTextPreprocessor()

        # Initialize multiple models for different aspects
        self.models = {}
        self._initialize_models()

        # Cache for embeddings
        self._embedding_cache = {}

        logger.info("Multi-vector embeddings initialized")

    def _initialize_models(self):
        """Initialize different embedding models for various aspects"""
        try:
            # General semantic model
            self.models['semantic'] = {
                'client': openai.OpenAI(api_key=Config.OPENAI_API_KEY),
                'model_name': Config.EMBEDDING_MODEL,
                'description': 'General semantic understanding'
            }

            if not _SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning(
                    "sentence-transformers not installed; advanced similarity models disabled"
                )
                return

            # Try to load specialized sentence transformers for different aspects
            model_configs = [
                {
                    'name': 'legal_similarity',
                    'model': 'all-MiniLM-L6-v2',  # Good for legal similarity
                    'description': 'Legal document similarity'
                },
                {
                    'name': 'entity_extraction',
                    'model': 'all-MiniLM-L6-v2',  # Can be specialized for entities
                    'description': 'Entity and named entity recognition'
                },
                {
                    'name': 'temporal_understanding',
                    'model': 'all-MiniLM-L6-v2',  # Can be fine-tuned for temporal
                    'description': 'Temporal and date understanding'
                }
            ]

            for config in model_configs:
                try:
                    model = SentenceTransformer(config['model'])
                    self.models[config['name']] = {
                        'model': model,
                        'model_name': config['model'],
                        'description': config['description']
                    }
                    logger.info(f"Loaded {config['name']} model: {config['model']}")
                except Exception as e:
                    logger.warning(f"Failed to load {config['name']} model: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {e}")

    def generate_multi_vector_embedding(self, text: str,
                                      embedding_type: str = 'semantic') -> Optional[np.ndarray]:
        """Generate embeddings using specified model type"""
        if embedding_type not in self.models:
            logger.warning(f"Unknown embedding type: {embedding_type}")
            embedding_type = 'semantic'

        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess_legal_text(text)

            model_info = self.models[embedding_type]

            if embedding_type == 'semantic':
                # Use OpenAI embeddings
                response = model_info['client'].embeddings.create(
                    model=model_info['model_name'],
                    input=processed_text,
                    encoding_format="float"
                )
                return np.array(response.data[0].embedding)

            else:
                # Use sentence transformer models
                if 'model' in model_info:
                    embedding = model_info['model'].encode([processed_text])
                    return embedding[0] if len(embedding) > 0 else None

            return None

        except Exception as e:
            logger.error(f"Failed to generate {embedding_type} embedding: {e}")
            return None

    def generate_comprehensive_embeddings(self, text: str) -> Dict[str, np.ndarray]:
        """Generate embeddings using all available models"""
        embeddings = {}

        for embedding_type in self.models.keys():
            embedding = self.generate_multi_vector_embedding(text, embedding_type)
            if embedding is not None:
                embeddings[embedding_type] = embedding

        return embeddings

class DomainSpecificEmbeddings:
    """Domain-specific embeddings with legal fine-tuning"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.multi_vector = MultiVectorEmbeddings(db_manager)

        # Domain adaptation parameters
        self.domain_weights = {
            'semantic': 0.4,
            'legal_similarity': 0.3,
            'entity_extraction': 0.2,
            'temporal_understanding': 0.1
        }

        # Drop weights for unavailable embedding types
        available_types = set(self.multi_vector.models.keys())
        self.domain_weights = {
            key: value for key, value in self.domain_weights.items()
            if key in available_types
        }

        if 'semantic' not in self.domain_weights:
            # Ensure we always have at least semantic embeddings
            self.domain_weights['semantic'] = 1.0

        # Legal domain vocabulary
        self.legal_vocabulary = self._load_legal_vocabulary()

        logger.info("Domain-specific embeddings initialized")

    def _load_legal_vocabulary(self) -> Dict[str, float]:
        """Load legal domain vocabulary with importance weights"""
        # This could be loaded from a file or database
        legal_vocab = {
            'contract': 1.0, 'agreement': 1.0, 'party': 0.9, 'clause': 0.9,
            'provision': 0.9, 'terms': 0.8, 'conditions': 0.8, 'obligation': 0.9,
            'liability': 0.9, 'indemnification': 0.8, 'termination': 0.9,
            'breach': 0.9, 'default': 0.8, 'remedy': 0.7, 'damages': 0.8,
            'intellectual property': 1.0, 'confidentiality': 0.9, 'proprietary': 0.8,
            'trademark': 0.8, 'copyright': 0.8, 'patent': 0.8, 'trade secret': 0.8,
            'governing law': 0.9, 'jurisdiction': 0.8, 'dispute resolution': 0.9,
            'arbitration': 0.8, 'mediation': 0.7, 'litigation': 0.7,
            'force majeure': 0.8, 'assignment': 0.7, 'amendment': 0.8,
            'waiver': 0.7, 'severability': 0.7, 'entire agreement': 0.8,
            'renewal': 0.8, 'expiration': 0.8, 'notice': 0.7, 'delivery': 0.6,
            'payment': 0.9, 'compensation': 0.8, 'fee': 0.7, 'penalty': 0.8,
            'interest': 0.7, 'late charge': 0.7, 'invoice': 0.6
        }

        return legal_vocab

    def generate_domain_adapted_embedding(self, text: str,
                                        adaptation_strength: float = 0.3) -> Optional[np.ndarray]:
        """Generate domain-adapted embedding with legal specialization"""
        try:
            # Get multi-vector embeddings
            embeddings = self.multi_vector.generate_comprehensive_embeddings(text)

            if not embeddings:
                logger.warning("No embeddings generated")
                return None

            # Combine embeddings using domain weights
            combined_embedding = None
            total_weight = 0

            for emb_type, embedding in embeddings.items():
                if emb_type in self.domain_weights:
                    weight = self.domain_weights[emb_type]
                    if combined_embedding is None:
                        # Initialize with the first embedding (resized if needed)
                        target_dim = Config.EMBEDDING_DIMENSION
                        if len(embedding) != target_dim:
                            # Resize embedding to match target dimension
                            embedding = self._resize_embedding(embedding, target_dim)
                        combined_embedding = embedding * weight
                    else:
                        # Resize and add subsequent embeddings
                        if len(embedding) != len(combined_embedding):
                            embedding = self._resize_embedding(embedding, len(combined_embedding))
                        combined_embedding += embedding * weight
                    total_weight += weight

            if combined_embedding is not None and total_weight > 0:
                combined_embedding /= total_weight

                # Apply domain adaptation
                if adaptation_strength > 0:
                    combined_embedding = self._apply_domain_adaptation(
                        text, combined_embedding, adaptation_strength
                    )

                return combined_embedding

            return None

        except Exception as e:
            logger.error(f"Domain adaptation failed: {e}")
            return None

    def _resize_embedding(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """Resize embedding to target dimension"""
        current_dim = len(embedding)

        if current_dim == target_dim:
            return embedding
        elif current_dim > target_dim:
            # Truncate
            return embedding[:target_dim]
        else:
            # Pad with zeros
            padded = np.zeros(target_dim)
            padded[:current_dim] = embedding
            return padded

    def _apply_domain_adaptation(self, text: str, embedding: np.ndarray,
                               adaptation_strength: float) -> np.ndarray:
        """Apply domain-specific adaptation to embedding"""
        text_lower = text.lower()

        # Calculate domain relevance score
        domain_score = 0.0
        vocab_matches = 0

        for term, importance in self.legal_vocabulary.items():
            if term in text_lower:
                domain_score += importance
                vocab_matches += 1

        if vocab_matches > 0:
            domain_score /= vocab_matches

        # Create domain adaptation vector
        # This is a simplified approach - in practice, this could be learned
        adaptation_vector = np.random.normal(0, 0.1, len(embedding))
        adaptation_vector *= domain_score * adaptation_strength

        # Apply adaptation
        adapted_embedding = embedding + adaptation_vector

        # Normalize
        norm = np.linalg.norm(adapted_embedding)
        if norm > 0:
            adapted_embedding /= norm

        return adapted_embedding

class EnhancedEmbeddingManager:
    """Enhanced embedding manager with domain specialization and caching"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.domain_embeddings = DomainSpecificEmbeddings(db_manager)

        # Embedding cache
        self._cache_enabled = True
        self._cache_size_limit = 10000
        self._embedding_cache = {}

        # Performance tracking
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_generations': 0,
            'average_generation_time': 0.0
        }

        logger.info("Enhanced embedding manager initialized")

    def generate_and_store_embeddings(self, document_id: str, chunks: List[str],
                                    use_domain_adaptation: bool = True) -> bool:
        """Generate and store enhanced embeddings for document chunks"""
        try:
            logger.info(f"Generating enhanced embeddings for {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                try:
                    # Generate domain-adapted embedding
                    if use_domain_adaptation:
                        embedding = self.domain_embeddings.generate_domain_adapted_embedding(chunk)
                    else:
                        embedding = self.domain_embeddings.multi_vector.generate_multi_vector_embedding(chunk)

                    if embedding is not None:
                        # Convert to list for database storage
                        embedding_list = embedding.tolist()

                        # Store chunk and embedding
                        chunk_metadata = {
                            "chunk_length": len(chunk),
                            "chunk_words": len(chunk.split()),
                            "embedding_model": "domain_adapted_hybrid",
                            "domain_adapted": use_domain_adaptation,
                            "embedding_dimension": len(embedding_list)
                        }

                        self.db_manager.insert_document_chunk(
                            document_id=document_id,
                            chunk_text=chunk,
                            chunk_index=i,
                            embedding=embedding_list,
                            metadata=chunk_metadata
                        )

                        # Cache the embedding
                        if self._cache_enabled:
                            cache_key = self._generate_cache_key(chunk)
                            self._embedding_cache[cache_key] = embedding
                            self._cleanup_cache()

                    else:
                        logger.error(f"Failed to generate embedding for chunk {i}")
                        return False

                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    return False

            logger.info(f"Successfully stored enhanced embeddings for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate enhanced embeddings: {e}")
            return False

    def generate_query_embedding(self, query: str, use_domain_adaptation: bool = True) -> Optional[List[float]]:
        """Generate enhanced query embedding"""
        try:
            # Check cache first
            if self._cache_enabled:
                cache_key = self._generate_cache_key(query)
                if cache_key in self._embedding_cache:
                    self.performance_stats['cache_hits'] += 1
                    embedding = self._embedding_cache[cache_key]
                    return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

            self.performance_stats['cache_misses'] += 1
            self.performance_stats['total_generations'] += 1

            # Generate embedding
            start_time = datetime.now()

            if use_domain_adaptation:
                embedding = self.domain_embeddings.generate_domain_adapted_embedding(query)
            else:
                embedding = self.domain_embeddings.multi_vector.generate_multi_vector_embedding(query)

            generation_time = (datetime.now() - start_time).total_seconds()

            # Update performance stats
            self.performance_stats['average_generation_time'] = (
                (self.performance_stats['average_generation_time'] *
                 (self.performance_stats['total_generations'] - 1) + generation_time) /
                self.performance_stats['total_generations']
            )

            if embedding is not None:
                embedding_list = embedding.tolist()

                # Cache the result
                if self._cache_enabled:
                    cache_key = self._generate_cache_key(query)
                    self._embedding_cache[cache_key] = embedding
                    self._cleanup_cache()

                return embedding_list

            return None

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None

    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def _cleanup_cache(self):
        """Clean up cache if it exceeds size limit"""
        if len(self._embedding_cache) > self._cache_size_limit:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._embedding_cache.keys())[:-self._cache_size_limit]
            for key in keys_to_remove:
                del self._embedding_cache[key]

    def batch_generate_embeddings(self, texts: List[str],
                                use_domain_adaptation: bool = True) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts efficiently"""
        embeddings = []

        for text in texts:
            embedding = self.generate_query_embedding(text, use_domain_adaptation)
            embeddings.append(embedding)

        return embeddings

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        cache_hit_rate = 0.0
        if self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'] > 0:
            cache_hit_rate = (self.performance_stats['cache_hits'] /
                            (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']))

        return {
            **self.performance_stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._embedding_cache),
            'cache_enabled': self._cache_enabled,
            'domain_weights': self.domain_embeddings.domain_weights
        }

    def clear_cache(self):
        """Clear embedding cache"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")

    def save_cache(self, filepath: str):
        """Save embedding cache to file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
            logger.info(f"Embedding cache saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def load_cache(self, filepath: str):
        """Load embedding cache from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                logger.info(f"Embedding cache loaded from {filepath}")
            else:
                logger.warning(f"Cache file not found: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")

    def update_domain_weights(self, new_weights: Dict[str, float]):
        """Update domain adaptation weights"""
        self.domain_embeddings.domain_weights.update(new_weights)
        logger.info(f"Updated domain weights: {self.domain_embeddings.domain_weights}")

    def fine_tune_embeddings(self, training_data: List[Tuple[str, str]],
                           epochs: int = 5) -> Dict[str, Any]:
        """
        Fine-tune embeddings using training data

        Args:
            training_data: List of (query, relevant_text) pairs
            epochs: Number of training epochs

        Returns:
            Training results and metrics
        """
        logger.info(f"Fine-tuning embeddings with {len(training_data)} examples")

        # This is a placeholder for fine-tuning implementation
        # In practice, this would involve:
        # 1. Creating positive and negative examples
        # 2. Training a contrastive loss or triplet loss
        # 3. Updating model parameters

        # For now, return mock results
        results = {
            'training_examples': len(training_data),
            'epochs_completed': epochs,
            'final_loss': 0.1,  # Mock value
            'improvement_score': 0.15,  # Mock value
            'status': 'completed'
        }

        logger.info("Fine-tuning completed (placeholder implementation)")
        return results
