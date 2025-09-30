"""
Smart Document Chunker for CLM automation system.
Implements semantic chunking with legal document structure awareness.
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """Types of document chunks"""
    HEADER = "header"
    SECTION = "section"
    CLAUSE = "clause"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    SIGNATURE = "signature"
    FOOTER = "footer"

@dataclass
class DocumentChunk:
    """Enhanced document chunk with metadata"""
    text: str
    chunk_type: ChunkType
    section_title: Optional[str]
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    importance_score: float = 0.0

class LegalDocumentParser:
    """Parser for legal document structure"""

    def __init__(self):
        # Legal document patterns
        self.section_patterns = [
            r'^(?:SECTION|SEC\.?|Article|ARTICLE)\s+\d+',
            r'^\d+\.\s+[A-Z][^\.]*[:\.]?$',
            r'^[A-Z][A-Z\s]{5,}:?\s*$',  # All caps headers
            r'^\w+\.\s+[A-Z]',  # Numbered sections
        ]

        self.clause_patterns = [
            r'^\([a-z]\)',  # (a), (b), (c)
            r'^\([0-9]+\)',  # (1), (2), (3)
            r'^\w+\.\w+',  # 1.1, 1.2, etc.
            r'^[a-z]\)',  # a), b), c)
        ]

        self.signature_patterns = [
            r'SIGNATURE|EXECUTED|SIGNED|WITNESS',
            r'By:\s*_+',
            r'Date:\s*_+',
            r'/s/',
        ]

        self.important_sections = [
            'termination', 'liability', 'indemnification', 'payment',
            'intellectual property', 'confidentiality', 'governing law',
            'dispute resolution', 'force majeure', 'amendment'
        ]

        logger.info("Legal document parser initialized")

    def identify_chunk_type(self, text: str, context: str = "") -> ChunkType:
        """Identify the type of document chunk"""
        text_clean = text.strip()

        # Check for signature blocks
        if any(re.search(pattern, text_clean, re.IGNORECASE)
               for pattern in self.signature_patterns):
            return ChunkType.SIGNATURE

        # Check for section headers
        if any(re.search(pattern, text_clean, re.MULTILINE)
               for pattern in self.section_patterns):
            return ChunkType.SECTION

        # Check for clauses
        if any(re.search(pattern, text_clean)
               for pattern in self.clause_patterns):
            return ChunkType.CLAUSE

        # Check for list items
        if re.match(r'^\s*[-•·]\s+', text_clean) or re.match(r'^\s*\d+\.\s+', text_clean):
            return ChunkType.LIST_ITEM

        # Check for tables (simple heuristic)
        if '\t' in text or '|' in text or text.count('  ') > 3:
            return ChunkType.TABLE

        # Default to paragraph
        return ChunkType.PARAGRAPH

    def extract_section_title(self, text: str) -> Optional[str]:
        """Extract section title from text"""
        lines = text.strip().split('\n')
        first_line = lines[0].strip()

        # Look for section headers
        for pattern in self.section_patterns:
            match = re.search(pattern, first_line, re.IGNORECASE)
            if match:
                # Clean up the title
                title = first_line.replace(match.group(), '').strip()
                return title if title else first_line

        # Check if first line looks like a title
        if (len(first_line) < 100 and
            first_line.isupper() or
            first_line.endswith(':') or
            not first_line.endswith('.')):
            return first_line

        return None

    def calculate_importance_score(self, text: str, chunk_type: ChunkType,
                                 section_title: str = None) -> float:
        """Calculate importance score for a chunk"""
        score = 0.0

        # Base score by chunk type
        type_scores = {
            ChunkType.SECTION: 0.8,
            ChunkType.CLAUSE: 0.7,
            ChunkType.PARAGRAPH: 0.5,
            ChunkType.LIST_ITEM: 0.4,
            ChunkType.SIGNATURE: 0.9,
            ChunkType.HEADER: 0.6,
            ChunkType.TABLE: 0.6,
            ChunkType.FOOTER: 0.2
        }
        score += type_scores.get(chunk_type, 0.5)

        # Boost for important legal terms
        text_lower = text.lower()
        for important_term in self.important_sections:
            if important_term in text_lower:
                score += 0.2

        # Boost for monetary amounts
        if re.search(r'\$[\d,]+', text) or re.search(r'\d+\s*(?:dollars?|USD)', text, re.IGNORECASE):
            score += 0.3

        # Boost for dates
        if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text):
            score += 0.2

        # Boost for parties/entities
        if re.search(r'\b(?:Corp|Corporation|Inc|LLC|Ltd|Company)\b', text, re.IGNORECASE):
            score += 0.2

        # Section title relevance
        if section_title:
            section_lower = section_title.lower()
            for important_term in self.important_sections:
                if important_term in section_lower:
                    score += 0.3

        return min(score, 1.0)  # Cap at 1.0

class SemanticChunker:
    """Semantic chunking using sentence similarity"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"Semantic chunker initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None

    def semantic_split(self, text: str, max_chunk_size: int = 1000,
                      similarity_threshold: float = 0.5) -> List[str]:
        """Split text into semantically coherent chunks"""
        if not self.sentence_model:
            # Fallback to simple splitting
            return self._simple_split(text, max_chunk_size)

        try:
            # Split into sentences
            sentences = self._split_sentences(text)
            if len(sentences) <= 1:
                return [text]

            # Generate sentence embeddings
            embeddings = self.sentence_model.encode(sentences)

            # Group sentences by semantic similarity
            chunks = []
            current_chunk = [sentences[0]]
            current_length = len(sentences[0])

            for i in range(1, len(sentences)):
                sentence = sentences[i]
                sentence_length = len(sentence)

                # Check if adding this sentence would exceed max size
                if current_length + sentence_length > max_chunk_size:
                    # Finalize current chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                    continue

                # Calculate similarity with current chunk
                chunk_embedding = np.mean([embeddings[j] for j in range(i-len(current_chunk), i)], axis=0)
                sentence_embedding = embeddings[i]

                similarity = cosine_similarity([chunk_embedding], [sentence_embedding])[0][0]

                if similarity >= similarity_threshold:
                    # Add to current chunk
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    # Start new chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length

            # Add final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks

        except Exception as e:
            logger.error(f"Semantic splitting failed: {e}")
            return self._simple_split(text, max_chunk_size)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with legal document awareness"""
        # Custom sentence splitting for legal documents
        sentences = []

        # Split by periods, but be careful with abbreviations
        parts = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)

        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

        return sentences

    def _simple_split(self, text: str, max_chunk_size: int) -> List[str]:
        """Simple text splitting fallback"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

class SmartDocumentChunker:
    """Main document chunker with legal awareness and semantic understanding"""

    def __init__(self, min_chunk_size: int = 200, max_chunk_size: int = 1000,
                 overlap_size: int = 100, semantic_threshold: float = 0.5):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.semantic_threshold = semantic_threshold

        self.legal_parser = LegalDocumentParser()
        self.semantic_chunker = SemanticChunker()

        logger.info(f"Smart document chunker initialized (min: {min_chunk_size}, max: {max_chunk_size})")

    def chunk_document(self, text: str, document_type: str = "contract") -> List[DocumentChunk]:
        """
        Chunk document with legal structure awareness and semantic coherence

        Args:
            text: Document text
            document_type: Type of document (contract, agreement, etc.)

        Returns:
            List of DocumentChunk objects
        """
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)

            # First, try to identify natural sections
            sections = self._identify_sections(cleaned_text)

            # Chunk each section
            all_chunks = []
            current_char = 0

            for section_idx, (section_text, section_title) in enumerate(sections):
                section_chunks = self._chunk_section(
                    section_text, section_title, current_char, section_idx
                )
                all_chunks.extend(section_chunks)
                current_char += len(section_text)

            # Add overlapping context between chunks
            all_chunks = self._add_overlap(all_chunks, text)

            logger.info(f"Document chunked into {len(all_chunks)} chunks")
            return all_chunks

        except Exception as e:
            logger.error(f"Document chunking failed: {e}")
            return self._fallback_chunking(text)

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess document text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')

        return text.strip()

    def _identify_sections(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """Identify major sections in the document"""
        sections = []
        lines = text.split('\n')

        current_section = []
        current_title = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line is a section header
            potential_title = self.legal_parser.extract_section_title(line)

            if potential_title and len(current_section) > 0:
                # Finalize previous section
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    sections.append((section_text, current_title))

                # Start new section
                current_section = [line]
                current_title = potential_title
            else:
                current_section.append(line)

        # Add final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append((section_text, current_title))

        return sections if sections else [(text, None)]

    def _chunk_section(self, section_text: str, section_title: Optional[str],
                      start_char: int, section_idx: int) -> List[DocumentChunk]:
        """Chunk a single section with semantic awareness"""
        if len(section_text) <= self.max_chunk_size:
            # Section is small enough to be a single chunk
            chunk_type = self.legal_parser.identify_chunk_type(section_text)
            importance_score = self.legal_parser.calculate_importance_score(
                section_text, chunk_type, section_title
            )

            return [DocumentChunk(
                text=section_text,
                chunk_type=chunk_type,
                section_title=section_title,
                chunk_index=0,
                start_char=start_char,
                end_char=start_char + len(section_text),
                metadata={
                    "section_index": section_idx,
                    "word_count": len(section_text.split()),
                    "char_count": len(section_text)
                },
                importance_score=importance_score
            )]

        # Use semantic chunking for larger sections
        semantic_chunks = self.semantic_chunker.semantic_split(
            section_text, self.max_chunk_size, self.semantic_threshold
        )

        chunks = []
        current_char = start_char

        for chunk_idx, chunk_text in enumerate(semantic_chunks):
            if len(chunk_text.strip()) < self.min_chunk_size:
                # Merge small chunks with the previous one
                if chunks:
                    chunks[-1].text += " " + chunk_text
                    chunks[-1].end_char = current_char + len(chunk_text)
                    chunks[-1].metadata["char_count"] = len(chunks[-1].text)
                    chunks[-1].metadata["word_count"] = len(chunks[-1].text.split())
                continue

            chunk_type = self.legal_parser.identify_chunk_type(chunk_text)
            importance_score = self.legal_parser.calculate_importance_score(
                chunk_text, chunk_type, section_title
            )

            chunk = DocumentChunk(
                text=chunk_text.strip(),
                chunk_type=chunk_type,
                section_title=section_title,
                chunk_index=chunk_idx,
                start_char=current_char,
                end_char=current_char + len(chunk_text),
                metadata={
                    "section_index": section_idx,
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text)
                },
                importance_score=importance_score
            )

            chunks.append(chunk)
            current_char += len(chunk_text)

        return chunks

    def _add_overlap(self, chunks: List[DocumentChunk], full_text: str) -> List[DocumentChunk]:
        """Add overlapping context between chunks"""
        if len(chunks) <= 1 or self.overlap_size <= 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            enhanced_text = chunk.text

            # Add overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                prev_words = prev_chunk.text.split()
                if len(prev_words) > 0:
                    overlap_words = prev_words[-min(self.overlap_size // 10, len(prev_words)):]
                    enhanced_text = ' '.join(overlap_words) + " " + enhanced_text

            # Add overlap from next chunk
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                next_words = next_chunk.text.split()
                if len(next_words) > 0:
                    overlap_words = next_words[:min(self.overlap_size // 10, len(next_words))]
                    enhanced_text = enhanced_text + " " + ' '.join(overlap_words)

            # Update chunk with enhanced text
            enhanced_chunk = DocumentChunk(
                text=enhanced_text,
                chunk_type=chunk.chunk_type,
                section_title=chunk.section_title,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "has_overlap": True,
                    "original_length": len(chunk.text),
                    "enhanced_length": len(enhanced_text)
                },
                importance_score=chunk.importance_score
            )

            overlapped_chunks.append(enhanced_chunk)

        return overlapped_chunks

    def _fallback_chunking(self, text: str) -> List[DocumentChunk]:
        """Simple fallback chunking when advanced methods fail"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for i, word in enumerate(words):
            word_length = len(word) + 1

            if current_length + word_length > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_type=ChunkType.PARAGRAPH,
                    section_title=None,
                    chunk_index=len(chunks),
                    start_char=0,  # Approximate
                    end_char=len(chunk_text),
                    metadata={"fallback": True, "word_count": len(current_chunk)},
                    importance_score=0.5
                )
                chunks.append(chunk)

                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = DocumentChunk(
                text=chunk_text,
                chunk_type=ChunkType.PARAGRAPH,
                section_title=None,
                chunk_index=len(chunks),
                start_char=0,
                end_char=len(chunk_text),
                metadata={"fallback": True, "word_count": len(current_chunk)},
                importance_score=0.5
            )
            chunks.append(chunk)

        return chunks

    def get_chunk_summary(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get summary statistics for chunks"""
        if not chunks:
            return {}

        total_chunks = len(chunks)
        total_chars = sum(len(chunk.text) for chunk in chunks)
        total_words = sum(chunk.metadata.get("word_count", 0) for chunk in chunks)

        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type.value
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

        avg_importance = sum(chunk.importance_score for chunk in chunks) / total_chunks

        return {
            "total_chunks": total_chunks,
            "total_characters": total_chars,
            "total_words": total_words,
            "average_chunk_size": total_chars // total_chunks,
            "chunk_types": chunk_types,
            "average_importance": avg_importance,
            "sections_identified": len(set(chunk.section_title for chunk in chunks if chunk.section_title))
        }