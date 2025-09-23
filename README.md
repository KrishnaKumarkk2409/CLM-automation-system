# 🤖 Contract Lifecycle Management (CLM) Automation System

![CLM System](https://img.shields.io/badge/CLM-Automation-blue) ![Python](https://img.shields.io/badge/Python-3.9+-green) ![LangChain](https://img.shields.io/badge/LangChain-Enabled-orange) ![Supabase](https://img.shields.io/badge/Supabase-Database-purple) ![Next.js](https://img.shields.io/badge/Next.js-14.0+-black) ![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)

> **AI-Powered Contract Intelligence Platform** - Streamlining contract management with cutting-edge AI automation, intelligent document processing, and proactive monitoring.

## 📋 Project Context

This system was developed as part of a technical assessment for **DeepRunner AI** - an enterprise AI automation company that empowers teams with AI agent solutions. The project demonstrates the implementation of a complete Contract Lifecycle Management platform using modern AI technologies and enterprise-grade architecture.

**🎯 Challenge**: Build an intelligent CLM platform that can automatically ingest, understand, alert, and provide insights on contract data across various departments.

**✨ Solution**: A comprehensive AI-powered system featuring RAG pipelines, LangChain agents, real-time monitoring, and both Streamlit and Next.js interfaces.

# 🚀 Quick Start

## Option 1: Docker (Recommended)

```bash
# 1. Clone repository
git clone <repository-url>
cd "CLM automation system"

# 2. Configure environment
cp docs/env.template.txt .env
# Edit .env with your actual values

# 3. Start system
./start-system.sh
```

## Option 2: Local Development

```bash
# 1. Clone repository
git clone <repository-url>
cd "CLM automation system"

# 2. Start development environment
./start-dev.sh
```

## Option 3: Manual Setup

### Prerequisites
- Docker & Docker Compose (for Option 1)
- Python 3.11+ & Node.js 18+ (for Option 2)
- OpenAI API key
- Supabase account

### Environment Configuration

1. **Copy template**: `cp docs/env.template.txt .env`
2. **Edit .env** with your values:

```env
# Required
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
OPENAI_API_KEY=your_openai_key

# Optional (for email reports)
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
REPORT_EMAIL=reports@example.com
```

3. **Database Setup**: Run the SQL schema from `database/schema.sql` in your Supabase project

## 🔍 System Access

- **Main Application**: http://localhost (via nginx)
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## ⚠️ Troubleshooting

**OCR Issues (Windows)**:
```bash
# Download and install Tesseract OCR from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

**macOS/Linux OCR**:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

For more details, see `docs/PROJECT_STRUCTURE.md`

## 📁 Project Structure

The project has been **recently reorganized** for better maintainability and scalability:

```
CLM automation system/
├── 📁 backend/              # FastAPI backend application
│   ├── app/                 # Main application code
│   └── src/                 # Source modules (config, database, etc.)
├── 📁 frontend/             # Next.js frontend application  
│   ├── src/                 # React components, pages, hooks
│   └── public/              # Static assets
├── 📁 database/             # Database schema and migrations
├── 📁 deployment/           # Docker & nginx configuration
│   ├── docker-compose.yml   # Container orchestration
│   └── nginx/               # Reverse proxy config
├── 📁 scripts/              # Automation & utility scripts
│   ├── start-system.sh      # Full Docker system startup
│   ├── start-dev.sh         # Development environment
│   └── generate_data.py     # Test data generation
├── 📁 docs/                 # Documentation & templates
│   ├── env.template.txt     # Environment variables template
│   └── PROJECT_STRUCTURE.md # Detailed structure guide
├── 📁 documents/            # Sample contracts (PDF, DOCX, TXT)
├── 📁 logs/                 # Application logs
└── 📁 uploads/              # Document upload storage
```

**🔗 Convenience Symlinks** (in root directory):
- `docker-compose.yml` → `deployment/docker-compose.yml`
- `start-system.sh` → `scripts/start-system.sh`
- `start-dev.sh` → `scripts/start-dev.sh`

> **Migration Note**: If upgrading from the old structure, your existing `.env` file will work without changes.

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

## 🏠 System Architecture

### Modern Multi-Interface Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Next.js UI      │    │   Streamlit     │    │  Daily Agent    │
│  (Production)    │    │   Chatbot       │    │  (Automated)    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └─────────────┬────────────────────────────┘
                      │
           ┌──────────┴──────────────────┐
           │      FastAPI Backend     │
           │  (REST API + WebSocket) │
           └──────────┬──────────────────┘
                      │
    ┌─────────────────┴─────────────────────┐
    │                   AI Layer                    │
    ├───────────────────────────────────────────┤
    │ RAG Pipeline | Embeddings | LangChain      │
    └─────────────────────┬─────────────────────┘
                       │
        ┌──────────────┴───────────────┐
        │   Document Processor    │
        │  (PDF/DOCX/TXT + OCR)   │
        └──────────────┬───────────────┘
                       │
        ┌──────────────┴───────────────┐
        │   Supabase Database     │
        │  (PostgreSQL + Vector)  │
        └─────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|----------|
| **Frontend** | Next.js 14, React, Tailwind CSS | Modern responsive UI with SSR |
| **Backend** | FastAPI, Python 3.11 | High-performance async API |
| **AI/ML** | OpenAI GPT-4, LangChain, RAG | Intelligent document processing |
| **Database** | Supabase (PostgreSQL + pgvector) | Vector storage & similarity search |
| **Infrastructure** | Docker, Nginx | Containerization & reverse proxy |
| **Monitoring** | Python logging, Email alerts | System health & notifications |

## 📋 Requirements

### Prerequisites
- Python 3.9+
- Supabase account and project
- OpenAI API key
- Email account for SMTP (optional, for reports)

### Dependencies

See `requirements.txt` for complete list.

## 📋 Installation & Setup

### 1. Clone & Environment Setup
```bash
git clone <repository-url>
cd "CLM automation system"
cp docs/env.template.txt .env  # Copy environment template
```

### 2. Choose Installation Method

#### Option A: Docker (Production-Ready)
```bash
# Configure .env file with your credentials
./start-system.sh
```

#### Option B: Development Setup
```bash
# Backend
cd backend && pip install -r requirements.txt

# Frontend  
cd frontend && npm install

# Start development environment
./start-dev.sh
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

## 📖 Usage Guide

### 🌐 Web Application (Recommended)

Access the modern Next.js interface:

- **Main App**: http://localhost (via nginx proxy)
- **Direct Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

**Key Features**:
- 💬 Interactive chat with contracts
- 📈 Analytics dashboard
- 📄 Document upload and processing
- ⚙️ System settings and configuration

### 📦 API Endpoints

**Document Management**:
```bash
# Upload contract
curl -X POST "http://localhost:8000/upload" \
  -F "file=@contract.pdf"

# Get all contracts
curl "http://localhost:8000/contracts"

# Search contracts  
curl "http://localhost:8000/search?query=termination+clause"
```

**AI-Powered Chat**:
```bash
# Ask questions about contracts
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Which contracts expire next month?"}'

# WebSocket chat (real-time)
# Connect to: ws://localhost:8000/ws/chat
```

**Analytics & Reports**:
```bash
# Get system analytics
curl "http://localhost:8000/analytics"

# Generate daily report
curl -X POST "http://localhost:8000/report"

# Check system health
curl "http://localhost:8000/health"
```

### 🔄 Data Management

**Generate Test Data**:
```bash
# Create sample contracts for testing
python scripts/generate_data.py
```

**Process Documents**:
```bash
# Via API
curl -X POST "http://localhost:8000/process" \
  -F "file=@documents/contract.pdf"

# Via CLI (backend directory)
cd backend && python -m app.main --process ../documents/
```

### 📊 Legacy Streamlit Interface

For backward compatibility:
```bash
cd backend && python -m app.main --chatbot
# Opens Streamlit interface at http://localhost:8501
```

## 🎯 System Components

### 📱 Frontend (`frontend/`)
- **Next.js 14**: Modern React framework with SSR
- **Tailwind CSS**: Utility-first responsive styling
- **React Query**: Efficient API state management
- **WebSocket Support**: Real-time chat functionality
- **Components**: Reusable UI elements and pages

### ⚙️ Backend (`backend/`)
- **FastAPI**: High-performance async API framework
- **WebSocket**: Real-time communication support
- **Document Processing**: Multi-format file handling with OCR
- **AI Integration**: RAG pipeline with LangChain & OpenAI
- **Database Management**: Supabase operations and vector search

### 📊 Core Modules (`backend/src/`)
- **Document Processor**: PDF, DOCX, TXT processing with OCR fallback
- **RAG Pipeline**: Vector search and context-aware responses
- **Contract Agent**: Automated monitoring and conflict detection
- **Database Manager**: Supabase integration with vector operations
- **Embeddings**: OpenAI embedding generation and storage

### 📦 Infrastructure (`deployment/`)
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Reverse proxy and load balancing
- **Environment Management**: Centralized configuration
- **SSL Support**: HTTPS certificate handling

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

## 👥 Development

### 🔧 Development Environment

**Start Development Mode**:
```bash
./start-dev.sh  # Starts both backend and frontend with hot reload
```

**Manual Development Setup**:
```bash
# Backend (Terminal 1)
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (Terminal 2)  
cd frontend
npm run dev
```

### 📝 Available Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| `./start-system.sh` | Production Docker setup | Root (symlink) |
| `./start-dev.sh` | Development environment | Root (symlink) |
| `scripts/generate_data.py` | Create test contracts | `scripts/` |

### 🧪 API Testing

**Interactive API Documentation**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

**Example API Calls**:
```bash
# Health check
curl http://localhost:8000/health

# Upload document
curl -X POST http://localhost:8000/upload \
  -F "file=@documents/sample.pdf"

# Chat with AI
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me expiring contracts"}'
```

## 🚀 Deployment

### 🐳 Production Deployment (Docker)

**Prerequisites**:
- Docker & Docker Compose
- Domain name (optional)
- SSL certificates (for HTTPS)

**Steps**:
```bash
# 1. Configure production environment
cp docs/env.template.txt .env
# Edit .env with production values

# 2. Deploy with Docker
./start-system.sh

# 3. Set up reverse proxy (nginx already included)
# 4. Configure SSL certificates in deployment/nginx/
```

### ⚙️ Environment Variables

**Required**:
```env
SUPABASE_URL=your_production_supabase_url
SUPABASE_KEY=your_production_supabase_key
OPENAI_API_KEY=your_openai_api_key
```

**Optional** (for email reports):
```env
EMAIL_USERNAME=your_email@company.com
EMAIL_PASSWORD=your_app_password
REPORT_EMAIL=reports@company.com
```

### 🔄 Automated Tasks

Set up cron jobs for automated contract monitoring:
```bash
# Daily contract monitoring at 9 AM
0 9 * * * curl -X POST http://localhost:8000/report

# Weekly system health check
0 1 * * 1 curl http://localhost:8000/health
```

### 🔍 Monitoring & Logs

**Log Locations**:
```bash
logs/clm_system.log    # Application logs
logs/errors.log        # Error tracking  
logs/access.log        # API access logs
```

**Health Monitoring**:
```bash
# Check system health
curl http://localhost:8000/health

# Monitor Docker containers
docker-compose -f deployment/docker-compose.yml ps

# View logs
docker-compose -f deployment/docker-compose.yml logs -f
```

---

## 🔗 Additional Resources

- **[Project Structure Guide](docs/PROJECT_STRUCTURE.md)** - Detailed architecture overview
- **[Environment Template](docs/env.template.txt)** - Configuration template
- **[API Documentation](http://localhost:8000/docs)** - Interactive API explorer
- **[Database Schema](database/schema.sql)** - Complete SQL schema

## 🤝 Contributing & Support

This project was developed as a technical demonstration for **DeepRunner AI**. The system showcases modern AI-powered document processing and contract lifecycle management capabilities.

### Key Achievements ✨
- ✅ **Full-stack Architecture**: React + FastAPI + Supabase
- ✅ **AI Integration**: RAG pipeline with OpenAI GPT-4
- ✅ **Document Processing**: Multi-format with OCR support  
- ✅ **Real-time Features**: WebSocket chat and notifications
- ✅ **Production Ready**: Docker deployment with nginx
- ✅ **Clean Architecture**: Modular, maintainable codebase

### Technical Highlights 🚀
- **Modern Stack**: Next.js 14, FastAPI, Supabase, Docker
- **AI/ML Pipeline**: LangChain, OpenAI embeddings, vector search
- **Enterprise Features**: Automated monitoring, conflict detection
- **Developer Experience**: Hot reload, API docs, comprehensive logging

---

**Built with ❤️ for DeepRunner AI** | 🚀 **Contract Intelligence Platform**

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