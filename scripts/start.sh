#!/bin/bash

# PAF Core Agent Development Start Script

echo "Starting PAF Core Agent..."

# Activate pyenv environment
echo "Activating pyenv environment..."
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate paf-core

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "
try:
    import sqlalchemy, alembic, asyncpg, apscheduler, fastapi
    print('✅ All dependencies are available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    print('Please run: pip install -r requirements.txt')
    exit(1)
"

# Install dependencies if requirements.txt is newer than last install
if [ requirements.txt -nt .last_install ] || [ ! -f .last_install ]; then
    echo "Installing/updating dependencies..."
    pip install -r requirements.txt
    touch .last_install
fi

# Set environment variables for development
export DEBUG=true
export HOST=0.0.0.0
export PORT=8000

# Start the application with hot reload
echo "Starting server on http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"
echo "Health check at http://localhost:8000/api/health"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 