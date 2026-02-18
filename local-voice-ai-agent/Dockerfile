# Dockerfile for Local Voice Chat Advanced + Party Manager
# Includes Kokoro TTS voices (kokoro-v1.0.onnx). No Blackhole - uses internal routing for AI Party.

FROM python:3.13-slim

WORKDIR /app

# Install system deps: git, curl (for model download), portaudio (for sounddevice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libportaudio2 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files first (faster rebuilds when only code changes)
COPY pyproject.toml ./
COPY . .

# Download Kokoro voice models (required for TTS with voice selection)
# voice_models.py looks for kokoro-v1.0.onnx and voices-v1.0.bin in /app
ARG KOKORO_BASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
RUN curl -fsSL -o kokoro-v1.0.onnx "${KOKORO_BASE}/kokoro-v1.0.onnx" \
    && curl -fsSL -o voices-v1.0.bin "${KOKORO_BASE}/voices-v1.0.bin" \
    && ls -la kokoro-v1.0.onnx voices-v1.0.bin

# Install Python deps (skip optional italian-stt for smaller image)
RUN pip install --no-cache-dir -e .

EXPOSE 7860

# Run the advanced voice chat app (includes Single Agent + AI Party with internal routing)
CMD ["python", "local_voice_chat_advanced.py", "--port", "7860"]
