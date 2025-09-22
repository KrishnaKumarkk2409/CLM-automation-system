"""
Embeddings module for CLM automation system.
Handles generation and management of document embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
import openai
from src.config import Config
from src.database import DatabaseManager

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages document embeddings using OpenAI"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
        # Initialize OpenAI client
        openai.api_key = Config.OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        
        logger.info("Embedding manager initialized with OpenAI")
    
    def generate_and_store_embeddings(self, document_id: str, chunks: List[str]) -> bool:
        """Generate embeddings for document chunks and store them"""
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks of document {document_id}")
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding for the chunk
                    embedding = self._generate_embedding(chunk)
                    
                    if embedding:
                        # Store chunk and embedding in database
                        chunk_metadata = {
                            "chunk_length": len(chunk),
                            "chunk_words": len(chunk.split()),
                            "embedding_model": Config.EMBEDDING_MODEL
                        }
                        
                        self.db_manager.insert_document_chunk(
                            document_id=document_id,
                            chunk_text=chunk,
                            chunk_index=i,
                            embedding=embedding,
                            metadata=chunk_metadata
                        )
                    else:
                        logger.error(f"Failed to generate embedding for chunk {i} of document {document_id}")
                        return False
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {i} for document {document_id}: {e}")
                    return False
            
            logger.info(f"Successfully generated and stored {len(chunks)} embeddings for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for document {document_id}: {e}")
            return False
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a text chunk using OpenAI"""
        try:
            # Clean text for embedding
            cleaned_text = self._clean_text_for_embedding(text)
            
            if not cleaned_text.strip():
                logger.warning("Empty text provided for embedding")
                return None
            
            # Generate embedding using OpenAI API
            response = self.client.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=cleaned_text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Validate embedding dimension
            if len(embedding) != Config.EMBEDDING_DIMENSION:
                logger.error(f"Unexpected embedding dimension: {len(embedding)} (expected {Config.EMBEDDING_DIMENSION})")
                return None
            
            return embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            return None
    
    def _clean_text_for_embedding(self, text: str) -> str:
        """Clean text before generating embeddings"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (OpenAI has token limits)
        max_tokens = 8000  # Conservative limit
        words = text.split()
        if len(words) > max_tokens:
            text = ' '.join(words[:max_tokens])
            logger.warning(f"Text truncated to {max_tokens} tokens for embedding")
        
        return text
    
    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for a query string"""
        return self._generate_embedding(query)
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts in batch"""
        embeddings = []
        
        for text in texts:
            embedding = self._generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings