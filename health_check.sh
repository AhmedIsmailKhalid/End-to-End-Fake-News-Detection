# Health check script for Docker container
set -e

# Check if FastAPI is responding
if ! curl -s -f "http://127.0.0.1:8000/docs" > /dev/null; then
    echo "FastAPI health check failed"
    exit 1
fi

# Check if Streamlit is responding
if ! curl -s -f "http://127.0.0.1:7860/_stcore/health" > /dev/null; then
    echo "Streamlit health check failed"
    exit 1
fi

# Check if required files exist
required_files=(
    "/tmp/model.pkl"
    "/tmp/vectorizer.pkl"
    "/tmp/data/combined_dataset.csv"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Required file missing: $file"
        exit 1
    fi
done

# Check if processes are running
if ! pgrep -f "schedule_tasks.py" > /dev/null; then
    echo "Scheduler process not running"
    exit 1
fi

echo "All health checks passed"
exit 0