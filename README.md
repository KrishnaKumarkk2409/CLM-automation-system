# 🤖 Contract Lifecycle Management (CLM) Automation System

![CLM System](https://img.shields.io/badge/CLM-Automation-blue) ![Python](https://img.shields.io/badge/Python-3.9+-green) ![LangChain](https://img.shields.io/badge/LangChain-Enabled-orange) ![Supabase](https://img.shields.io/badge/Supabase-Database-purple) ![Next.js](https://img.shields.io/badge/Next.js-14.0+-black) ![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)

> **AI-Powered Contract Intelligence Platform** - Streamlining contract management with cutting-edge AI automation, intelligent document processing, and proactive monitoring.

## 📋 Project Context

This system was developed as part of a technical assessment for **DeepRunner AI** - an enterprise AI automation company that empowers teams with AI agent solutions. The project demonstrates the implementation of a complete Contract Lifecycle Management platform using modern AI technologies and enterprise-grade architecture.

**🎯 Challenge**: Build an intelligent CLM platform that can automatically ingest, understand, alert, and provide insights on contract data across various departments.

**✨ Solution**: A comprehensive AI-powered system featuring RAG pipelines, LangChain agents, real-time monitoring, and both Streamlit and Next.js interfaces.

# Quick Start

Steps:-

1. Clone Repo
2. Paste the .env.template file provided in the email thread
3. Configure env and Run setup.py file

if Mac/linux: 
```bash
python3 -m venv env

source env/bin/activate

python setup.py 
```
if Windows:

```bash
python -m venv env

./env/Scripts/activate

python setup.py
```
 OCR Processing Issues may arrise if you are in windows OS.
```
Error: OCR extraction failed
```
**Solution**: Install Tesseract OCR system package:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

4. python main.py --chatbot
```bash 
# To See streamlit UI of the raw code.  
```

Thanks.


## 🌟 Features

### Core Capabilities
- **📄 Document Ingestion**: Process PDF, DOCX, and TXT files with OCR support for scanned documents
- **🤖 RAG Pipeline**: Advanced retrieval-augmented generation for contract queries
- **🔍 Semantic Search**: Find similar documents and contracts using vector embeddings
- **⚠️ Conflict Detection**: Automatically identify inconsistencies between contracts
- **📅 Expiration Monitoring**: Track and alert on upcoming contract expirations
- **📊 Daily Reports**: Automated daily monitoring and email alerts
- **💬 AI Chatbot**: Interactive Streamlit interface for contract queries
- **🔗 Vector Database**: Supabase with pgvector for efficient similarity search

### AI-Powered Features
- **Smart Contract Analysis**: Extract parties, dates, clauses, and contact information
- **Conflict Resolution**: Detect address mismatches, date conflicts, and data inconsistencies
- **Natural Language Queries**: Ask questions about contracts in plain English
- **Source Citation**: All responses include references to source documents
- **Document Similarity**: Find contracts with similar terms or structures

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   CLI Interface │    │  Daily Agent    │
│   Chatbot       │    │   (main.py)     │    │  (Scheduled)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────┬───────────────────────────────┘
                         │
          ┌─────────────┴───────────┐
          │   RAG Pipeline          │
          │  (LangChain + OpenAI)   │
          └─────────────┬───────────┘
                        │
          ┌─────────────┴───────────┐
          │   Embedding Manager     │
          │  (OpenAI Embeddings)    │
          └─────────────┬───────────┘
                        │
          ┌─────────────┴───────────┐
          │   Document Processor    │
          │  (PDF/DOCX/TXT + OCR)   │
          └─────────────┬───────────┘
                        │
          ┌─────────────┴───────────┐
          │   Supabase Database     │
          │  (PostgreSQL + Vector)  │
          └─────────────────────────┘
```

## 📋 Requirements

### Prerequisites
- Python 3.9+
- Supabase account and project
- OpenAI API key
- Email account for SMTP (optional, for reports)

### Dependencies

See `requirements.txt` for complete list.

## 🚀 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd "CLM automation system"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Supabase Database [Utilise .env provided on email , db already configured.]


Execute the provided SQL schema in your Supabase project:

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

### 4. Configure Environment [Provided in Email ]
Copy `.env.template` to `.env` and fill in your credentials:

```env
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
REPORT_EMAIL=recipient@company.com

# Application Configuration
DOCUMENTS_FOLDER=./documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_THRESHOLD=0.7
```

## 📖 Usage

### 1. Generate Sample Data
```bash
python src/generate_synthetic_data.py
```

This creates 15 realistic contract documents in various formats with intentional variations and conflicts for testing.

### 2. Process Documents
```bash
# Process all documents in the documents folder
python main.py --process

# Process documents from a specific folder
python main.py --process --folder /path/to/contracts
```

### 3. Interactive Mode
```bash
# Start interactive CLI
python main.py --interactive

# Available commands in interactive mode:
CLM> help
CLM> process
CLM> What contracts expire in the next 30 days?
CLM> similar software license agreement
CLM> find TechCorp
CLM> report
CLM> monitor
```

### 4. Streamlit Chatbot
```bash
# Launch web interface
python main.py --chatbot
# or
streamlit run src/chatbot_interface.py
```

### 5. Command Line Queries
```bash
# Ask questions about contracts
python main.py --query "What are the payment terms in the TechCorp contract?"

# Generate daily report
python main.py --report

# Run monitoring cycle
python main.py --monitor

# Find similar documents
python main.py --similar "maintenance agreement"
python main.py --find "TechCorp_Software_License_2024.pdf"
```

## 🎯 System Components

### Document Processor (`src/document_processor.py`)
- **File Support**: PDF, DOCX, TXT
- **OCR Capability**: Handles scanned PDFs using Tesseract
- **Text Chunking**: Intelligent document segmentation
- **Metadata Extraction**: Automatic contract information parsing

### RAG Pipeline (`src/rag_pipeline.py`)
- **Vector Search**: Supabase-powered similarity matching
- **Context Preparation**: Smart chunk selection and formatting
- **LLM Integration**: GPT-4 powered responses with source citations
- **Query Processing**: Natural language understanding

### Contract Agent (`src/contract_agent.py`)
- **LangChain Agents**: Tool-equipped AI for complex tasks
- **Daily Monitoring**: Automated contract health checks
- **Conflict Detection**: Advanced pattern matching for inconsistencies
- **Report Generation**: Professional formatted reports
- **Email Integration**: SMTP-based alert system

### Database Manager (`src/database.py`)
- **Supabase Integration**: Full CRUD operations
- **Vector Operations**: Embedding storage and similarity search
- **Conflict Analysis**: Cross-contract comparison logic
- **Performance Optimized**: Indexed queries and batch operations

### Chatbot Interface (`src/chatbot_interface.py`)
- **Interactive UI**: Streamlit-powered web interface
- **Real-time Chat**: Conversational contract queries
- **Visualizations**: Contract timelines and department charts
- **System Controls**: Document processing and report generation
- **Source Citations**: Transparent response attribution

## 🔍 Key Features in Detail

### Document Similarity
The system can find similar documents using semantic search:

```python
# Find documents similar to a text query
similar_docs = rag_pipeline.find_similar_contracts("software licensing terms")

# Find documents similar to an existing contract
similar_docs = db_manager.get_similar_documents(document_id)
```

### Conflict Detection
Automatically identifies:
- **Contact Information Mismatches**: Different addresses, emails, or phones for the same company
- **Date Conflicts**: Inconsistent end dates or renewal terms
- **Party Name Variations**: Same company referenced differently

### Smart Contract Analysis
Extracts structured information:
- Contract names and types
- Party identification
- Key dates (start, end, renewal)
- Financial terms and amounts
- Important clauses and provisions
- Contact information

### Natural Language Queries
Ask questions like:
- "Which contracts expire in the next 60 days?"
- "What are the termination clauses in the DataSystems contract?"
- "Show me all contracts with CloudVentures Corp"
- "Find conflicts between TechCorp agreements"

## 🛡️ Error Handling & Logging

The system includes comprehensive logging and error handling:

### Logging Levels
- **DEBUG**: Detailed function calls and data flow
- **INFO**: General operations and status updates
- **WARNING**: Non-critical issues and fallbacks
- **ERROR**: Failures and exceptions with stack traces
- **CRITICAL**: System-level failures

### Log Files
- `./logs/clm_system.log` - Main application log
- `./logs/errors.log` - Error-specific log
- `./logs/chatbot.log` - Chatbot interface log

### Error Tracking
- Unique error IDs for tracking
- Error frequency monitoring
- Performance metrics logging
- Health check reporting

## 🚨 Troubleshooting

### Common Issues

#### 1. Database Connection Errors
```
Error: Failed to initialize Supabase client
```
**Solution**: Check your `SUPABASE_URL` and `SUPABASE_KEY` in `.env`

#### 2. OpenAI API Errors
```
Error: OpenAI embedding generation failed
```
**Solution**: Verify `OPENAI_API_KEY` and check API quota

#### 3. OCR Processing Issues
```
Error: OCR extraction failed
```
**Solution**: Install Tesseract OCR system package:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### 4. Memory Issues with Large Documents
```
Error: Memory allocation failed
```
**Solution**: 
- Reduce `CHUNK_SIZE` in configuration
- Process documents in smaller batches
- Monitor system resources

### Performance Optimization

1. **Vector Search Tuning**:
   - Adjust `SIMILARITY_THRESHOLD` (0.6-0.8 range)
   - Optimize chunk sizes for your document types
   - Use appropriate embedding models

2. **Database Optimization**:
   - Regular index maintenance
   - Archive old documents
   - Monitor query performance

3. **Resource Management**:
   - Monitor API rate limits
   - Implement caching for frequent queries
   - Use batch processing for large datasets

## 🔐 Security Considerations

### Data Protection
- Environment variables for sensitive credentials
- Database-level access controls
- Encrypted data transmission
- Audit logging for all operations

### API Security
- Rate limiting on OpenAI API calls
- Input validation and sanitization
- Error message sanitization
- Secure credential storage

### Access Control
- Role-based database permissions
- Logging of all access attempts
- Secure email transmission
- Data retention policies

## 📈 Monitoring and Analytics

### Performance Metrics
- Document processing speed
- Query response times
- Embedding generation rates
- Error rates by operation

### Business Metrics
- Contract expiration tracking
- Conflict detection rates
- Department utilization
- Query pattern analysis

### Health Monitoring
- System resource usage
- Database performance
- API quota consumption
- Log file sizes

## 🚀 Deployment

### Production Deployment
1. Set up production Supabase instance
2. Configure production OpenAI API keys
3. Set up secure SMTP for email reports
4. Implement backup and disaster recovery
5. Set up monitoring and alerting

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py", "--interactive"]
```

### Scheduled Tasks
Set up cron jobs for daily monitoring:
```bash
# Daily report at 9 AM
0 9 * * * cd /path/to/clm && python main.py --monitor

# Weekly document processing
0 1 * * 1 cd /path/to/clm && python main.py --process
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all functions
- Include error handling
- Write comprehensive tests

## 📄 License

[Add your license information here]

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review log files in `./logs/`
- Open an issue on GitHub
- Contact system administrator

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: Document processing in multiple languages
- **Advanced Analytics**: Machine learning for contract risk assessment
- **Integration APIs**: REST API for external system integration
- **Mobile App**: React Native mobile interface
- **Blockchain Integration**: Smart contract verification
- **Advanced OCR**: AI-powered document understanding

### Roadmap
- Q1: Enhanced conflict detection algorithms
- Q2: REST API and webhook support
- Q3: Advanced analytics dashboard  
- Q4: Multi-tenant architecture

---

**CLM Automation System** - Streamlining contract management with AI-powered automation.

*Built with ❤️ using LangChain, OpenAI, and Supabase*