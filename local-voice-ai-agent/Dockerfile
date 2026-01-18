# ============================================================================
# Stage 1: Builder - Build FFmpeg 7 and install dependencies with uv
# ============================================================================
FROM python:3.13-slim-bookworm AS builder

# Install build dependencies for PyAV and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    wget \
    xz-utils \
    nasm \
    yasm \
    cmake \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libmp3lame-dev \
    libopus-dev \
    libvorbis-dev \
    libtheora-dev \
    zlib1g-dev \
    libaom-dev \
    && rm -rf /var/lib/apt/lists/*

# Build FFmpeg 7 from source
WORKDIR /tmp
RUN wget -q https://ffmpeg.org/releases/ffmpeg-7.0.2.tar.xz && \
    tar -xf ffmpeg-7.0.2.tar.xz && \
    cd ffmpeg-7.0.2 && \
    ./configure \
        --prefix=/usr/local \
        --enable-gpl \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libvpx \
        --enable-libmp3lame \
        --enable-libopus \
        --enable-libvorbis \
        --enable-libtheora \
        --enable-libaom \
        --enable-shared \
        --disable-static \
        --disable-doc \
    && make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/ffmpeg*

# Update library path for FFmpeg
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Create virtual environment and sync dependencies
RUN uv venv /app/.venv && \
    uv sync --frozen --no-dev

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.13-slim-bookworm AS runtime

# Install runtime dependencies (audio libs + FFmpeg runtime libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libsndfile1 \
    pulseaudio-utils \
    curl \
    libx264-164 \
    libx265-199 \
    libvpx7 \
    libmp3lame0 \
    libopus0 \
    libvorbis0a \
    libvorbisenc2 \
    libtheora0 \
    libaom3 \
    && rm -rf /var/lib/apt/lists/*

# Copy FFmpeg from builder
COPY --from=builder /usr/local/bin/ff* /usr/local/bin/
COPY --from=builder /usr/local/lib/libav* /usr/local/lib/
COPY --from=builder /usr/local/lib/libsw* /usr/local/lib/
COPY --from=builder /usr/local/lib/libpostproc* /usr/local/lib/

# Update library cache
RUN ldconfig

# Create non-root user
RUN groupadd --gid 1000 voiceai && \
    useradd --uid 1000 --gid 1000 --create-home voiceai

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY --chown=voiceai:voiceai . .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOCKER_CONTAINER=true \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Create cache directories with correct ownership for volume mounts
RUN mkdir -p /home/voiceai/.cache/huggingface \
             /home/voiceai/.cache/kokoro && \
    chown -R voiceai:voiceai /home/voiceai/.cache

# Switch to non-root user
USER voiceai

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "local_voice_chat_advanced.py"]
