FROM python:3.11.6-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    curl \
    wget \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/data /tmp/model /tmp/logs /app/logs && \
    chmod -R 755 /tmp/data /tmp/model /tmp/logs /app/logs

# Make scripts executable
RUN chmod +x /app/start.sh

# Copy initial datasets if they exist
RUN if [ -f /app/data/combined_dataset.csv ]; then \
        cp /app/data/combined_dataset.csv /tmp/data/; \
    fi

# Initialize system
RUN python /app/initialize_system.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/health_check.sh

# Change ownership to appuser
RUN chown -R appuser:appuser /app /tmp/data /tmp/model /tmp/logs

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 7860 8000

# Run the startup script
CMD ["./start.sh"]