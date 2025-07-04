#!/bin/bash

# Robust startup script with error handling and health checks
set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Function to check if port is available
check_port() {
    local port=$1
    if netstat -tuln | grep -q ":$port "; then
        warning "Port $port is already in use"
        return 1
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    log "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            log "$service_name is ready!"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        ((attempt++))
    done
    
    error "$service_name failed to start within expected time"
    return 1
}

# Function to handle shutdown gracefully
cleanup() {
    log "Shutting down services gracefully..."
    
    # Kill background processes
    if [ ! -z "$SCHEDULER_PID" ]; then
        kill $SCHEDULER_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$FASTAPI_PID" ]; then
        kill $FASTAPI_PID 2>/dev/null || true
    fi
    
    # Wait for processes to terminate
    wait
    
    log "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main startup sequence
main() {
    log "Starting Fake News Detection System..."
    
    # Create necessary directories
    mkdir -p /tmp/data /tmp/model /tmp/logs
    
    # Initialize system if needed
    if [ ! -f "/tmp/model.pkl" ] || [ ! -f "/tmp/vectorizer.pkl" ]; then
        log "Initializing system..."
        python /app/initialize_system.py
        if [ $? -ne 0 ]; then
            error "System initialization failed"
            exit 1
        fi
    fi
    
    # Check required files
    required_files=(
        "/tmp/data/combined_dataset.csv"
        "/tmp/model.pkl"
        "/tmp/vectorizer.pkl"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            warning "Required file missing: $file"
            log "Attempting to create missing file..."
            python /app/initialize_system.py
            break
        fi
    done
    
    # Start FastAPI server
    log "Starting FastAPI server..."
    uvicorn app.fastapi_server:app \
        --host 127.0.0.1 \
        --port 8000 \
        --log-level info \
        --access-log \
        --workers 1 &
    
    FASTAPI_PID=$!
    
    # Wait for FastAPI to be ready
    if ! wait_for_service "FastAPI" "http://127.0.0.1:8000/docs"; then
        error "FastAPI failed to start"
        exit 1
    fi
    
    # Start background services
    log "Starting scheduler..."
    python scheduler/schedule_tasks.py &> /tmp/scheduler.log &
    SCHEDULER_PID=$!
    
    log "Starting drift monitor..."
    python monitor/monitor_drift.py &> /tmp/monitor.log &
    MONITOR_PID=$!
    
    # Start Streamlit (foreground)
    log "Starting Streamlit interface..."
    exec streamlit run app/streamlit_app.py \
        --server.port=7860 \
        --server.address=0.0.0.0 \
        --server.enableCORS false \
        --server.enableXsrfProtection false \
        --server.maxUploadSize 10 \
        --server.enableStaticServing true \
        --logger.level info
}

# Run main function
main "$@"