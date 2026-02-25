FROM python:3.13-slim-trixie

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl openssl \
    ffmpeg \
    pkg-config \
    build-essential \
    python3-dev \
    cython3 \
    libportaudio2 portaudio19-dev \
    libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev \
    libswresample-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 7860
CMD ["uv", "run", "python", "local_voice_chat_advanced.py", "--port", "7860"]