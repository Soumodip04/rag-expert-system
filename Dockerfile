FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data models chroma_db_general chroma_db_healthcare chroma_db_legal chroma_db_financial

# Expose ports
EXPOSE 8501 8000

# Default command (can be overridden)
CMD ["python", "test_complete_system.py"]
