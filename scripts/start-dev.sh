#!/bin/bash

# CLM Automation System Development Startup Script
echo "🚀 Starting CLM Automation System in development mode..."

# Create necessary directories
mkdir -p logs uploads

# Check if .env file exists
if [ ! -f .env ]; then
  echo "⚠️  Warning: .env file not found. Creating from template..."
  if [ -f .env.template ]; then
    cp .env.template .env
    echo "✅ Created .env file from template. Please edit with your credentials."
  else
    echo "❌ Error: .env.template not found."
    echo "Please create a .env file manually with required environment variables."
  fi
fi

# Start backend in background
echo "🔧 Starting backend server..."
cd backend
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to initialize
sleep 2

# Start frontend
echo "🎨 Starting frontend development server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Function to handle script termination
cleanup() {
  echo "🛑 Stopping development servers..."
  kill $BACKEND_PID $FRONTEND_PID
  echo "👋 Development environment stopped"
  exit 0
}

# Register the cleanup function for script termination
trap cleanup SIGINT SIGTERM

echo "✅ Development environment started!"
echo "📊 Frontend: http://localhost:3000"
echo "🔌 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Keep script running
wait