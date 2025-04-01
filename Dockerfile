# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-mac/requirements.txt .

# Install torch CPU version first to ensure we don't get GPU version
RUN pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Create required directories
RUN mkdir -p uploads outputs

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=5512

# Copy .env file
COPY .env .

# Expose port
EXPOSE 5512

# Run the application
CMD ["python", "api_server.py"]
