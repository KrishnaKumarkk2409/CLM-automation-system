-- Create proper RLS policies for the CLM system
-- Run these in your Supabase SQL editor

-- Create policies for documents table
CREATE POLICY "Allow service role all access" ON documents
FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow authenticated users to read documents" ON documents
FOR SELECT USING (auth.role() = 'authenticated');

-- Create policies for document_chunks table
CREATE POLICY "Allow service role all access" ON document_chunks
FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow authenticated users to read chunks" ON document_chunks
FOR SELECT USING (auth.role() = 'authenticated');

-- Create policies for contracts table
CREATE POLICY "Allow service role all access" ON contracts
FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow authenticated users to read contracts" ON contracts
FOR SELECT USING (auth.role() = 'authenticated');