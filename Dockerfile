FROM python:3.11.6-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 7860

# Run both FastAPI and Streamlit using a wrapper script
# CMD ["python", "app.py"]
# CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
CMD ["bash", "-c", "python scheduler/schedule_tasks.py & python monitor/monitor_drift.py & streamlit run app/streamlit_app.py"]

