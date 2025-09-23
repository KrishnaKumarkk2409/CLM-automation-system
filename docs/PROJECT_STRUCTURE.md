# CLM Automation System - Project Structure

## Directory Layout

```
CLM automation system/
├── 📁 backend/                     # Backend FastAPI application
│   ├── app/
│   │   └── main.py                 # FastAPI main application
│   ├── src/                        # Core source code modules
│   │   ├── config.py              # Configuration management
│   │   ├── database.py            # Database operations
│   │   ├── embeddings.py          # OpenAI embeddings
│   │   ├── rag_pipeline.py        # RAG implementation
│   │   ├── contract_agent.py      # AI contract agent
│   │   ├── document_processor.py  # Document processing
│   │   └── logging_config.py      # Logging configuration
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                 # Backend container config
│
├── 📁 frontend/                    # Frontend Next.js application
│   ├── src/
│   │   ├── app/                   # Next.js app directory
│   │   │   ├── layout.tsx         # App layout
│   │   │   ├── page.tsx           # Main page
│   │   │   └── api/health/        # Health check API
│   │   ├── components/            # React components
│   │   │   ├── chat/              # Chat interface
│   │   │   ├── analytics/         # Analytics dashboard
│   │   │   ├── upload/            # Document upload
│   │   │   └── layout/            # Layout components
│   │   └── hooks/                 # Custom React hooks
│   ├── package.json               # Node.js dependencies
│   ├── tailwind.config.js         # Tailwind CSS config
│   └── Dockerfile                 # Frontend container config
│
├── 📁 database/                    # Database schema and migrations
│   └── schema.sql                 # Supabase database schema
│
├── 📁 deployment/                  # Deployment configuration
│   ├── docker-compose.yml         # Container orchestration
│   └── nginx.conf                 # Reverse proxy config
│
├── 📁 scripts/                     # Utility scripts
│   ├── start-system.sh           # Production startup script
│   ├── start-dev.sh              # Development startup script
│   └── test-system.py            # System test script
│
├── 📁 docs/                        # Documentation
│   ├── env.template.txt          # Environment variables template
│   └── PROJECT_STRUCTURE.md      # This file
│
├── 📁 documents/                   # Sample contract documents
│   ├── *.pdf                     # PDF contracts (5 files)
│   ├── *.docx                    # Word documents (4 files)
│   └── *.txt                     # Text files (5 files)
│
├── 📁 logs/                        # Application logs
├── 📁 uploads/                     # Uploaded documents
├── 📁 ssl/                         # SSL certificates (for production)
│
├── .env                           # Environment variables (create from template)
├── .gitignore                     # Git ignore patterns
├── README.md                      # Main documentation
│
└── Symlinks (for convenience):
    ├── docker-compose.yml -> deployment/docker-compose.yml
    ├── start-system.sh -> scripts/start-system.sh
    ├── start-dev.sh -> scripts/start-dev.sh
    ├── test-system.py -> scripts/test-system.py
    └── env.template.txt -> docs/env.template.txt
```

## Key Components

### Backend (`/backend`)
- **FastAPI Application**: Modern async Python web framework
- **Core Modules**: Modular architecture for maintainability
- **Document Processing**: Support for PDF, DOCX, TXT with OCR
- **AI Pipeline**: RAG implementation with LangChain
- **Database**: Supabase integration with vector storage

### Frontend (`/frontend`)
- **Next.js 14**: React framework with TypeScript
- **Components**: Modular React components
- **Real-time Chat**: WebSocket support for live updates
- **Analytics**: Interactive dashboards and visualizations
- **Modern UI**: Tailwind CSS with responsive design

### Deployment (`/deployment`)
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Reverse proxy and load balancing
- **Production Ready**: Health checks and monitoring

### Scripts (`/scripts`)
- **Development**: Local development with auto-reload
- **Production**: Docker-based deployment
- **Testing**: Automated system validation

## Quick Start Commands

```bash
# Development mode (local)
./start-dev.sh

# Production mode (Docker)
./start-system.sh

# Run tests
./test-system.py

# Docker compose (manual)
cd deployment && docker-compose up --build
```

## Environment Setup

1. Copy environment template: `cp docs/env.template.txt .env`
2. Edit `.env` with your actual values
3. Run setup script based on your preferred mode

## File Naming Conventions

- **Scripts**: `kebab-case.sh` or `kebab-case.py`
- **Components**: `PascalCase.tsx`
- **Utilities**: `snake_case.py`
- **Config**: `lowercase.config.js`

## Development Workflow

1. **Local Development**: Use `./start-dev.sh` for rapid iteration
2. **Testing**: Run `./test-system.py` to validate changes
3. **Production Testing**: Use `./start-system.sh` to test Docker setup
4. **Deployment**: Use `deployment/docker-compose.yml` for production

This structure promotes:
- 📦 **Modularity**: Clear separation of concerns
- 🔧 **Maintainability**: Organized code structure
- 🚀 **Scalability**: Docker-ready deployment
- 👥 **Team Collaboration**: Consistent organization
- 📚 **Documentation**: Self-documenting structure