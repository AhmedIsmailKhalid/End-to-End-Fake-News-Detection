#!/bin/bash

# Start background scripts
python scheduler/schedule_tasks.py &> logs/scheduler.log &
python monitor/monitor_drift.py &> logs/monitor.log &

# Start Streamlit in the foreground
exec streamlit run app/streamlit_app.py --server.port=7860 --server.address=0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
