FROM python:3.9-slim

# Install system dependencies (FAISS-CPU requires libgomp1 for OpenMP compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python package dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY static/ ./static/
COPY backend/ ./backend/

# Create persistent storage directories
RUN mkdir -p data/uploads data/vectorstore

# Expose API port
EXPOSE 8000

# Prevent python from buffering logs
ENV PYTHONUNBUFFERED=1

# Startup command
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
