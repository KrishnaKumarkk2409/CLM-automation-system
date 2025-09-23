# 🚀 CLM Automation System - Quick Start Guide

This guide will get your Contract Lifecycle Management system up and running in minutes.

## 📋 Prerequisites

- **Python 3.9+** (Check with: `python --version`)
- **Supabase Account** (Free tier available: [supabase.com](https://supabase.com))
- **OpenAI API Key** (Get from: [platform.openai.com](https://platform.openai.com))
- **Email Account** (Optional, for daily reports)

## ⚡ Quick Installation

### 1. **Automated Setup** (Recommended)
```bash
# Run the automated setup script
python setup.py

# Follow the prompts to:
# - Install dependencies
# - Create .env file
# - Generate sample data
# - Verify installation
```

### 2. **Manual Setup** (Alternative)
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.template .env

# Edit .env file with your credentials
nano .env  # or use your preferred editor

# Generate sample contract data
python src/generate_synthetic_data.py

# Test the system
python test_system.py
```

## 🔧 Configuration

Edit your `.env` file with the following:

```env
# Supabase Configuration (Required)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# OpenAI Configuration (Required)
OPENAI_API_KEY=sk-your-openai-api-key

# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
REPORT_EMAIL=recipient@company.com
```

### 🗄️ Supabase Database Setup

1. **Create a new Supabase project** at [supabase.com](https://supabase.com)
2. **Go to SQL Editor** in your dashboard
3. **Run this SQL script** to create the required tables:

```sql
-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create document_chunks table for RAG
CREATE TABLE document_chunks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(1536), -- OpenAI embedding dimension
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create contracts table for structured data
CREATE TABLE contracts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    contract_name TEXT,
    parties JSONB,
    start_date DATE,
    end_date DATE,
    renewal_date DATE,
    status TEXT DEFAULT 'active',
    key_clauses JSONB,
    contact_info JSONB,
    department TEXT,
    conflicts JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create similarity search function
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1536),
    match_threshold float,
    match_count int
) RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_text TEXT,
    similarity float
) LANGUAGE SQL STABLE AS $$
    SELECT 
        document_chunks.id,
        document_chunks.document_id,
        document_chunks.chunk_text,
        1 - (document_chunks.embedding <=> query_embedding) AS similarity
    FROM document_chunks
    WHERE 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY document_chunks.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Create indexes
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON contracts (end_date);
CREATE INDEX ON contracts (renewal_date);
CREATE INDEX ON documents (file_type);
```

4. **Copy your Project URL and Anon Key** from Settings > API

## 🎮 Usage

### **Web Interface** (Recommended)
```bash
python main.py --chatbot
```
- Opens at `http://localhost:8501`
- Interactive chat with your contracts
- Visual dashboards and analytics
- Document processing controls

### **Command Line Interface**
```bash
# Process contracts (first time)
python main.py --process

# Ask questions
python main.py --query "Which contracts expire soon?"

# Generate daily report
python main.py --report

# Find similar documents
python main.py --similar "software licensing"

# Interactive mode
python main.py --interactive
```

### **Scheduled Monitoring**
```bash
# Set up daily monitoring (crontab)
0 9 * * * cd /path/to/clm && python main.py --monitor
```

## 🧪 Testing

Verify everything works:
```bash
python test_system.py
```

This will test:
- ✅ All imports and dependencies
- ✅ Configuration validation  
- ✅ Database connectivity
- ✅ Sample data generation
- ✅ System readiness

## 📄 Sample Data

The system includes 15 realistic contract documents:
- **5 PDFs**: Including scanned contracts with OCR
- **4 DOCX**: Draft agreements and amendments
- **3 TXT**: Summaries and email correspondence  
- **2 Unstructured**: Meeting notes and discussions

**Key Features:**
- Intentional conflicts for testing
- Various expiration dates
- Multiple departments (IT, Legal, Finance, etc.)
- Version variations of the same contracts

## 🚨 Common Issues

### **Import Errors**
```bash
# Update packages to latest versions
pip install --upgrade -r requirements.txt
```

### **Database Connection Failed**
- Check your Supabase URL and key in `.env`
- Ensure vector extension is enabled
- Verify tables were created correctly

### **OpenAI API Errors**
- Verify your API key is correct
- Check your OpenAI account has credits
- Ensure the key has proper permissions

### **OCR Not Working**
```bash
# macOS
brew install tesseract

# Ubuntu/Debian  
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## 🎯 Example Queries

Try these in the chatbot or CLI:

```bash
"What contracts expire in the next 30 days?"
"Show me all TechCorp agreements"
"What are the payment terms in our contracts?"
"Find conflicts between CloudVentures contracts"
"Which departments have the most contracts?"
"What's the total value of all active contracts?"
```

## 🔧 Advanced Configuration

### **Performance Tuning**
```env
# Adjust these in .env
CHUNK_SIZE=1000           # Text chunk size
CHUNK_OVERLAP=200         # Overlap between chunks
SIMILARITY_THRESHOLD=0.7  # Vector search threshold
```

### **Email Reports**
For automated daily reports:
1. Enable 2-factor auth on Gmail
2. Generate an App Password  
3. Use the app password in `EMAIL_PASSWORD`

### **Custom Documents**
Place your contract files in the `./documents` folder:
- Supported formats: PDF, DOCX, TXT
- OCR automatically handles scanned PDFs
- Metadata is extracted automatically

## 📈 Next Steps

Once everything is running:

1. **Process Your Contracts**: Replace sample data with real contracts
2. **Set Up Monitoring**: Configure daily email reports
3. **Explore Features**: Try the web interface and different query types
4. **Customize**: Adjust configuration for your specific needs

## 🆘 Getting Help

- **Test Issues**: Run `python test_system.py` for diagnostics
- **Documentation**: See `README.md` for comprehensive guide
- **Logs**: Check `./logs/` folder for error details
- **Configuration**: Verify `.env` file settings

## 🎉 Success!

If the test script shows all green checkmarks, you're ready to:

```bash
# Start the web interface
python main.py --chatbot

# Or use the interactive CLI
python main.py --interactive
```

Your AI-powered Contract Lifecycle Management system is now operational! 🚀

---
**Need more help?** Check the full documentation in `README.md` or run the diagnostic test script.