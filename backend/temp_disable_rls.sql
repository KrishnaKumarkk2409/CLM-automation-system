-- Temporary SQL commands to disable RLS for testing
-- Run these in your Supabase SQL editor

-- Disable RLS on documents table
ALTER TABLE documents DISABLE ROW LEVEL SECURITY;

-- Disable RLS on document_chunks table  
ALTER TABLE document_chunks DISABLE ROW LEVEL SECURITY;

-- Disable RLS on contracts table
ALTER TABLE contracts DISABLE ROW LEVEL SECURITY;

-- To re-enable later (for production), use:
-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE contracts ENABLE ROW LEVEL SECURITY;