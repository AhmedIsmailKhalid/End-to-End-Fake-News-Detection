#!/bin/bash

# Start background jobs
python scheduler/schedule_tasks.py &> logs/scheduler.log &
python monitor/monitor_drift.py &> logs/monitor.log &

# Start FastAPI (internal port)
uvicorn app.fastapi_server:app --host 127.0.0.1 --port 8000 &

# Start Streamlit (foreground)
exec streamlit run app/streamlit_app.py \
  --server.port=7860 \
  --server.address=0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
