"""
Database operations module for CLM automation system.
Handles Supabase connections and database operations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
import json
from supabase import create_client, Client
from src.config import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations with Supabase"""
    
    def __init__(self):
        """Initialize Supabase client"""
        try:
            self.client: Client = create_client(
                Config.SUPABASE_URL,
                Config.SUPABASE_KEY
            )
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    def insert_document(self, filename: str, file_type: str, content: str, metadata: Dict[str, Any]) -> str:
        """Insert a document into the documents table"""
        try:
            result = self.client.table('documents').insert({
                'filename': filename,
                'file_type': file_type,
                'content': content,
                'metadata': metadata,
                'updated_at': datetime.now().isoformat()
            }).execute()
            
            document_id = result.data[0]['id']
            logger.info(f"Document {filename} inserted with ID: {document_id}")
            return document_id
        except Exception as e:
            logger.error(f"Failed to insert document {filename}: {e}")
            raise
    
    def insert_document_chunk(self, document_id: str, chunk_text: str, 
                            chunk_index: int, embedding: List[float], 
                            metadata: Dict[str, Any] = None) -> str:
        """Insert a document chunk with embedding"""
        try:
            result = self.client.table('document_chunks').insert({
                'document_id': document_id,
                'chunk_text': chunk_text,
                'chunk_index': chunk_index,
                'embedding': embedding,
                'metadata': metadata or {}
            }).execute()
            
            chunk_id = result.data[0]['id']
            logger.info(f"Document chunk {chunk_index} inserted for document {document_id}")
            return chunk_id
        except Exception as e:
            logger.error(f"Failed to insert document chunk: {e}")
            raise
    
    def insert_contract(self, document_id: str, contract_data: Dict[str, Any]) -> str:
        """Insert contract structured data"""
        try:
            result = self.client.table('contracts').insert({
                'document_id': document_id,
                'contract_name': contract_data.get('contract_name'),
                'parties': contract_data.get('parties', []),
                'start_date': contract_data.get('start_date'),
                'end_date': contract_data.get('end_date'),
                'renewal_date': contract_data.get('renewal_date'),
                'status': contract_data.get('status', 'active'),
                'key_clauses': contract_data.get('key_clauses', []),
                'contact_info': contract_data.get('contact_info', {}),
                'department': contract_data.get('department'),
                'conflicts': contract_data.get('conflicts', []),
                'updated_at': datetime.now().isoformat()
            }).execute()
            
            contract_id = result.data[0]['id']
            logger.info(f"Contract inserted with ID: {contract_id}")
            return contract_id
        except Exception as e:
            logger.error(f"Failed to insert contract: {e}")
            raise
    
    def similarity_search(self, query_embedding: List[float], 
                         threshold: float = Config.SIMILARITY_THRESHOLD, 
                         limit: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search using the database function"""
        try:
            result = self.client.rpc('match_documents', {
                'query_embedding': query_embedding,
                'match_threshold': threshold,
                'match_count': limit
            }).execute()
            
            return result.data
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise
    
    def get_expiring_contracts(self, days: int = Config.EXPIRATION_WARNING_DAYS) -> List[Dict[str, Any]]:
        """Get contracts expiring within specified days"""
        try:
            target_date = (datetime.now().date() + 
                          timedelta(days=days)).isoformat()
            
            result = self.client.table('contracts')\
                .select('*, documents!inner(filename)')\
                .lte('end_date', target_date)\
                .gte('end_date', datetime.now().date().isoformat())\
                .eq('status', 'active')\
                .execute()
            
            return result.data
        except Exception as e:
            logger.error(f"Failed to get expiring contracts: {e}")
            raise
    
    def find_contract_conflicts(self) -> List[Dict[str, Any]]:
        """Find contracts with potential conflicts"""
        try:
            # Get all active contracts with their documents
            result = self.client.table('contracts')\
                .select('*, documents!inner(filename)')\
                .eq('status', 'active')\
                .execute()
            
            contracts = result.data
            conflicts = []
            
            # Check for conflicts between contracts
            for i, contract1 in enumerate(contracts):
                for contract2 in contracts[i+1:]:
                    conflict = self._detect_conflict(contract1, contract2)
                    if conflict:
                        conflicts.append(conflict)
            
            return conflicts
        except Exception as e:
            logger.error(f"Failed to find contract conflicts: {e}")
            raise
    
    def _detect_conflict(self, contract1: Dict, contract2: Dict) -> Optional[Dict[str, Any]]:
        """Detect conflicts between two contracts"""
        conflicts = []
        
        # Check for same parties with different contact info
        if (contract1.get('parties') and contract2.get('parties') and
            self._parties_overlap(contract1['parties'], contract2['parties'])):
            
            # Check contact info conflicts
            if (contract1.get('contact_info') and contract2.get('contact_info')):
                contact_conflicts = self._compare_contact_info(
                    contract1['contact_info'], contract2['contact_info']
                )
                if contact_conflicts:
                    conflicts.extend(contact_conflicts)
            
            # Check date conflicts for overlapping periods
            date_conflicts = self._check_date_conflicts(contract1, contract2)
            if date_conflicts:
                conflicts.extend(date_conflicts)
        
        if conflicts:
            return {
                'contract1': {
                    'id': contract1['id'],
                    'filename': contract1['documents']['filename'],
                    'contract_name': contract1.get('contract_name', 'Unnamed')
                },
                'contract2': {
                    'id': contract2['id'],
                    'filename': contract2['documents']['filename'],
                    'contract_name': contract2.get('contract_name', 'Unnamed')
                },
                'conflicts': conflicts,
                'detected_at': datetime.now().isoformat()
            }
        
        return None
    
    def _parties_overlap(self, parties1: List[str], parties2: List[str]) -> bool:
        """Check if two contracts have overlapping parties"""
        return bool(set(parties1) & set(parties2))
    
    def _compare_contact_info(self, contact1: Dict, contact2: Dict) -> List[Dict]:
        """Compare contact information for conflicts"""
        conflicts = []
        
        for key in ['email', 'phone', 'address']:
            if (key in contact1 and key in contact2 and 
                contact1[key] != contact2[key]):
                conflicts.append({
                    'type': 'contact_info_mismatch',
                    'field': key,
                    'value1': contact1[key],
                    'value2': contact2[key]
                })
        
        return conflicts
    
    def _check_date_conflicts(self, contract1: Dict, contract2: Dict) -> List[Dict]:
        """Check for date-related conflicts"""
        conflicts = []
        
        # Check for different end dates for same parties
        if (contract1.get('end_date') and contract2.get('end_date') and
            contract1['end_date'] != contract2['end_date']):
            conflicts.append({
                'type': 'end_date_mismatch',
                'date1': contract1['end_date'],
                'date2': contract2['end_date']
            })
        
        return conflicts
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        try:
            result = self.client.table('documents')\
                .select('*')\
                .eq('id', document_id)\
                .execute()
            
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def get_similar_documents(self, document_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to the given document"""
        try:
            # Get the document's first chunk embedding
            chunk_result = self.client.table('document_chunks')\
                .select('embedding')\
                .eq('document_id', document_id)\
                .eq('chunk_index', 0)\
                .execute()
            
            if not chunk_result.data:
                return []
            
            embedding = chunk_result.data[0]['embedding']
            
            # Find similar documents
            similar = self.similarity_search(embedding, limit=limit + 1)
            
            # Filter out the original document and get unique documents
            seen_docs = set()
            results = []
            
            for item in similar:
                if item['document_id'] != document_id and item['document_id'] not in seen_docs:
                    doc = self.get_document_by_id(item['document_id'])
                    if doc:
                        doc['similarity'] = item['similarity']
                        results.append(doc)
                        seen_docs.add(item['document_id'])
                        
                        if len(results) >= limit:
                            break
            
            return results
        except Exception as e:
            logger.error(f"Failed to find similar documents: {e}")
            return []
    
    def update_contract_conflicts(self, contract_id: str, conflicts: List[Dict]) -> bool:
        """Update contract with detected conflicts"""
        try:
            result = self.client.table('contracts')\
                .update({'conflicts': conflicts, 'updated_at': datetime.now().isoformat()})\
                .eq('id', contract_id)\
                .execute()
            
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Failed to update contract conflicts: {e}")
            return False