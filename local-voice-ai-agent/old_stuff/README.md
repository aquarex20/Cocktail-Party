# Local Voice AI Agent

A real-time voice chat application powered by local AI models. This project allows you to have voice conversations with AI models like Gemma running locally on your machine.

## Features

- Real-time speech-to-text conversion
- Local LLM inference using Ollama
- Text-to-speech response generation
- Web interface for interaction
- Phone number interface option

## Prerequisites

- MacOS
- [Ollama](https://ollama.ai/) - Run LLMs locally
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

## Installation

### 1. Install prerequisites with Homebrew

```bash
brew install ollama
brew install uv
```

### 2. Clone the repository

```bash
git clone https://github.com/jesuscopado/local-voice-ai-agent.git
cd local-voice-ai-agent
```

### 3. Set up Python environment and install dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

**Optional – Italian speech-to-text:** If you use the Language setting **Italian**, install the optional Whisper-based STT so the app can transcribe Italian speech:

```bash
uv sync --extra italian-stt
```

Or with pip: `pip install ".[italian-stt]"` (installs `torch`, `torchaudio`, and `transformers`).  
The first time you use Italian, the Whisper model (~967MB) is downloaded once and stored under `local-voice-ai-agent/.cache/whisper`; later runs reuse it and do not re-download.

### 4. Download required models in Ollama

```bash
ollama pull gemma3:1b
# For advanced version
ollama pull gemma3:4b
```

## Usage

### Basic Voice Chat

```bash
python local_voice_chat.py
```

### Advanced Voice Chat (with system prompt)

#### Web UI (default)
```bash
python local_voice_chat_advanced.py
```

#### Phone Number Interface
Get a temporary phone number that anyone can call to interact with your AI:
```bash
python local_voice_chat_advanced.py --phone
```

### Audio device configuration (optional)

You can choose which audio input and output devices to use instead of relying on system defaults.

- **Without options**: Uses your system default input/output (e.g. built-in mic and speakers).
- **With options**: Set devices explicitly for the “party” setup (BlackHole + Script Config).

Examples:
```bash
# Use BlackHole 2ch as input and device index 1 as output
python local_voice_chat_advanced.py --input-device "BlackHole 2ch" --output-device 1

# Testing mode: lists devices, explains config, can run local_party.py, then starts session
python local_voice_chat_advanced.py --testing

# Testing with devices pre-set
python local_voice_chat_advanced.py --input-device "BlackHole 2ch" --output-device "Your Headphones" --testing
```

**Testing mode** (`--testing`): Runs a short setup flow that (1) explains how `local_party.py` and `local_voice_chat_advanced.py` share audio, (2) lists available input/output devices, (3) lets you type input/output device names or indices, (4) optionally starts `local_party.py` in the background so you can test that you hear both the script and the AI on the same output, and (5) reminds you to check system sound settings if something is wrong. After you press Enter, the voice chat session starts.

**How the two apps fit together:**

- **local_party.py** (Script Player) outputs to **Script Config** (a multi-output device that sends to **BlackHole 2ch** and to your **headphones/speakers**).
- **local_voice_chat_advanced.py** takes **BlackHole 2ch** as *input* and your **headphones/speakers** as *output*.

Use the **same output device** (your headphones/speakers) for both so you hear the script and the AI in one place. If you can’t hear properly, check:

1. System sound output is set to your headphones/speakers.
2. In Audio MIDI Setup, **Script Config** includes that same device.
3. This app’s `--output-device` is set to that same device.

### Blackhole config for party

Open **Audio MIDI Setup** on Mac (or equivalent multi-output device settings elsewhere). Create a new **Multi-Output Device**. In that device, select as outputs: **BlackHole 2ch** and your **headphones** (or desired output). Name it **Script Config** (or anything you like, and update `sd.default.device` in `local_party.py` to match). Set your Mac’s sound output to **Script Config**. You can comment out the `sd.default.device` lines in both files if you prefer to use system default for testing.

### AI Party: multiple AIs talking to each other

The app includes an **AI Party** tab that runs 2–4 AI agents conversing with each other via **internal audio routing** (no Blackhole or virtual devices):

- Each agent: STT → LLM → TTS
- Agent A's TTS output is fed directly to Agent B's "mic" (and vice versa) via an in-process `AudioBus`
- Docker-friendly, works on any platform
- Optional: monitor all agents to your headphones

### Docker (optional)

A `Dockerfile` and `docker-compose.yml` are provided for running the web UI in a container. Full audio personalization (Blackhole, voice chat) requires macOS; use the native app for AI Party.

```bash
docker compose up --build
```

## How it works

The application uses:
- `FastRTC` for WebRTC communication
- `Moonshine` for local speech-to-text conversion
- `Kokoro` for text-to-speech synthesis
- `Ollama` for running local LLM inference with `Gemma` models

When you speak, your audio is:
1. Transcribed to text using Moonshine
2. Sent to a local LLM via Ollama for processing
3. The LLM response is converted back to speech with Kokoro
4. The audio response is streamed back to you via FastRTC
