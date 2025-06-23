FROM python:3.11.6-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will use
EXPOSE 7860

# Set environment variable to suppress Streamlit warnings
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run FastAPI server in background, then launch Streamlit
CMD uvicorn app.fastapi_server:app --host 0.0.0.0 --port 8000 & \
    streamlit run app/streamlit_app.py
