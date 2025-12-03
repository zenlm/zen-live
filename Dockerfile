FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV, audio, and streaming
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    ffmpeg \
    libasound2 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8000
EXPOSE 9000/udp
EXPOSE 1935

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Default: run the application
# Override with: docker run ... pytest tests/
CMD ["python", "app.py"]
