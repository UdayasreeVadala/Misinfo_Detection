FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (Docker layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY misinfo_env.py .
COPY server.py .
COPY inference.py .
COPY openenv.yaml .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Health check so HF Space knows when server is ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/ping').raise_for_status()"

# Start the FastAPI server — note: lowercase server (matches server.py filename)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]