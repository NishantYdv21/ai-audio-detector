# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run the app with gunicorn (required for Cloud Run)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "backend.main:app", "-k", "uvicorn.workers.UvicornWorker"]
