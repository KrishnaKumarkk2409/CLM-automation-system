#!/bin/bash

# Development startup script for CLM Automation System
echo "🚀 Starting CLM Automation System (Development Mode)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Check environment variables
echo -e "${BLUE}🔐 Checking environment configuration...${NC}"

if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found, creating from template...${NC}"
    cat > .env << EOL
# CLM Automation System Environment Variables

# Required - Supabase Database
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here

# Required - OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Optional - Email Configuration
EMAIL_USERNAME=your_email@domain.com
EMAIL_PASSWORD=your_app_password_here
REPORT_EMAIL=reports@domain.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Optional - Application Settings
DOCUMENTS_FOLDER=./documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_THRESHOLD=0.7

# Development URLs
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
EOL
    echo -e "${YELLOW}📝 Please edit .env file with your actual configuration${NC}"
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p logs uploads documents

# Install backend dependencies
echo -e "${BLUE}📦 Installing backend dependencies...${NC}"
cd backend
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}🔧 Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

cd ..

# Install frontend dependencies
echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"
cd frontend
npm install
cd ..

echo -e "${GREEN}✅ Installation complete!${NC}"

# Start services
echo -e "${BLUE}🚀 Starting services...${NC}"

# Start backend
echo -e "${BLUE}Starting FastAPI backend...${NC}"
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo -e "${BLUE}Starting Next.js frontend...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}🎉 CLM Automation System is starting!${NC}"
echo -e "${BLUE}📊 Backend API: http://localhost:8000${NC}"
echo -e "${BLUE}🌐 Frontend: http://localhost:3000${NC}"
echo -e "${BLUE}📚 API Docs: http://localhost:8000/docs${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down services...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}👋 CLM Automation System stopped${NC}"
    exit 0
}

# Trap CTRL+C and cleanup
trap cleanup INT

# Wait for processes
wait