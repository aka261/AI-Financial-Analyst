# Dockerfile
# ----------
# Packages the Corporate Financial Analyst application into a container.
# Built by Google Cloud Build and stored in Artifact Registry.

# Use Python 3.11 slim image to keep container size small
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
# If requirements.txt does not change, this layer is cached
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY financial_parser.py .
COPY analyst_agent.py .
COPY server.py .
COPY mcp_server.py .
COPY static/ ./static/

# Create required directories
RUN mkdir -p uploads watched reports rag_db

# Expose the application port
EXPOSE 5000

# Health check so Kubernetes knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Run the application
CMD ["python", "server.py"]