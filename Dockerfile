FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV, audio, and streaming
# Note: libgl1-mesa-glx was deprecated in Debian Trixie, use libgl1 instead
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    # FFmpeg for SRT/RTMP/NDI streaming
    ffmpeg \
    # Audio dependencies
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
# SRT input/output (UDP)
EXPOSE 9000/udp
# RTMP input/output
EXPOSE 1935

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Run the application
CMD ["python", "app.py"]
