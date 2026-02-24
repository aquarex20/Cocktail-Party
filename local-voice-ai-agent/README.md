# CocktailPartyAI — Local Voice Agent (with diarization refinement)

Real-time voice chat using local models (STT → LLM → TTS) plus an optional **diarization server** that assigns speaker labels (e.g. `SPK_00`, `SPK_01`) and runs a background **refinement loop** to improve speaker assignments over time.

## Capabilities

- **Real-time voice chat UI**: WebRTC audio in/out with a Gradio web UI (`voice_chat.py`)
- **Local LLM**: runs via Ollama (default: `gemma3:4b`)
- **Speech-to-text (STT)**: WhisperX-based transcription in `whisper_stt.py`
- **Text-to-speech (TTS)**: Kokoro streaming output
- **Optional diarization server**: FastAPI server (`diarization_server.py`) that
  - diarizes each utterance and assigns speakers to words when transcript segments are provided
  - keeps a rolling audio buffer per session
  - periodically **re-runs diarization over recent audio** and updates stored utterances with `assigned_refined`

## Prerequisites

- Python **3.11+**
- [uv](https://github.com/astral-sh/uv) for dependency management
- [Ollama](https://ollama.ai/) for running the local LLM
- Optional: an NVIDIA GPU (CUDA) if you want GPU acceleration for diarization / WhisperX

## Installation

From `Cocktail-Party/local-voice-ai-agent/`:

```bash
brew install uv ollama
uv venv
source .venv/bin/activate
uv sync
```

This folder expects the Kokoro weights to be present:
- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

Download the LLM in Ollama:

```bash
ollama pull gemma3:4b
```

## Configuration

Create a local `.env` (do not commit it):

```bash
cp .env.example .env
```

- **Hugging Face token (required for diarization)**: WhisperX diarization uses pyannote models hosted on Hugging Face; you must set `HUGGING_FACE_TOKEN` (or `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`) in `.env` for diarization to work.
- **CPU/GPU tuning**:
  - `DIARIZATION_DEVICE=auto|cpu|cuda`
  - `WHISPERX_DEVICE=auto|cpu|cuda|mps`
  - `WHISPERX_COMPUTE_TYPE=int8|float16|float32`
  - `WHISPERX_MODEL=small|medium|...`

## Running

### 1) Start the diarization server (optional but recommended)

```bash
source .venv/bin/activate
uvicorn diarization_server:app --host 127.0.0.1 --port 8001
```

Notes:
- If the Hugging Face token is missing, diarization requests will fail and the server will store an error for the affected utterances.
- The refinement loop runs automatically on server startup (see env vars below).

### 2) Start the voice chat UI

```bash
source .venv/bin/activate
python voice_chat.py
```

The UI posts utterances to the diarization server via `DIARIZATION_SERVER_URL` (defaults to `http://127.0.0.1:8001`).

## How it works (high level)

1. The browser streams audio via WebRTC.
2. `ReplyOnPause` detects an end-of-turn pause and triggers `response(...)`.
3. `whisper_stt.py` transcribes the accumulated audio (WhisperX).
4. The transcript is sent to the local LLM via Ollama.
5. The answer is streamed back as audio using Kokoro TTS.
6. In parallel, `voice_chat.py` sends audio + transcript segments to `diarization_server.py` (fire-and-forget).
7. The server stores `assigned` speaker labels for the utterance, then periodically refines recent history and updates `assigned_refined`.

## Diarization refinement loop (server)

`diarization_server.py` keeps a rolling audio buffer per session and runs a background thread that:
- re-diarizes the most recent window of audio
- remaps speakers to stable “global” speaker IDs
- re-assigns word speakers for stored utterances that overlap the refine window (`assigned_refined`)

Environment variables:
- `DIARIZATION_REFINE_INTERVAL_S` (default `10`): how often refinement runs
- `DIARIZATION_REFINE_MIN_S` (default `30`): minimum buffered audio required before refining
- `DIARIZATION_REFINE_MAX_S` (default `120`): rolling window size for refinement
- `DIARIZATION_STORE_MAX_S` (default `300`): how much audio to keep per session

## Machine-dependent performance tuning (CPU vs GPU)

Some defaults are chosen for broad compatibility (often CPU-friendly). Depending on your machine, you may want to tune:

- **WhisperX STT** (`whisper_stt.py`):
  - `WHISPERX_MODEL` (bigger = better accuracy, slower)
  - `WHISPERX_DEVICE` and `WHISPERX_COMPUTE_TYPE` (e.g. `cuda + float16` for GPUs, `cpu + int8` for CPUs)
  - `WHISPER_MIN_AUDIO_S` to control how short utterances are ignored
- **Diarization server** (`diarization_server.py`):
  - `DIARIZATION_DEVICE` to force CPU or GPU
  - refinement window settings (`DIARIZATION_REFINE_*`) to trade latency/compute for higher-quality speaker assignments

## Legacy / experimental scripts

- `voice_chat_legacy.py`: older UI version kept for reference.
- `old_stuff/`: older prototypes and experiments (not kept in sync with the current entrypoints).

## Next steps (future improvements): adaptive pause detection / turn-taking

Right now, the “when should the AI answer?” behavior is mainly controlled by **pause detection**: `FastRTC`’s `ReplyOnPause` accumulates audio and triggers your reply callback when it decides the user has paused/stopped speaking.

In real cocktail-party settings (noise, cross-talk, interruptions, fast back-and-forth), a single fixed configuration is rarely optimal. A strong next step is to **tune these parameters dynamically during a conversation** based on:
- background noise level / mic gain
- whether multiple speakers are talking over each other
- interaction style (“quick banter” vs “thoughtful long answers”)
- how often the user gets “cut off” vs how often the AI feels sluggish to respond

Below are the two configuration classes (as defined in `fastrtc`) that govern this behavior.

### `AlgoOptions` (pause-detection algorithm settings)

Defined in `fastrtc/reply_on_pause.py`.

- **`audio_chunk_duration` (float, seconds)**: how much audio must accumulate before the VAD check is run on that buffer. Larger values generally mean the system **waits longer** before concluding “pause detected”, at the cost of latency.
- **`started_talking_threshold` (float, seconds)**: if the VAD estimates more than this much speech inside the current chunk, the user is considered to have **started talking** (helps avoid triggering on tiny noises).
- **`speech_threshold` (float, seconds)**: after “started talking” is true, if the VAD estimates *less* speech than this inside the chunk, it is treated as **stopped speaking / pause detected** (smaller values usually make it less eager to stop).
- **`max_continuous_speech_s` (float, seconds)**: a safety cap; if continuous speech reaches this duration, the handler triggers even if a pause is not detected (useful to prevent never-ending utterances).

### `SileroVadOptions` (Silero VAD model settings)

Defined in `fastrtc/pause_detection/silero.py`. These settings affect how raw audio is labeled as speech vs silence.

- **`threshold` (float)**: Silero outputs a speech probability per window; probabilities **above** this are treated as speech. Higher = stricter (can miss quiet speech); lower = more sensitive (can treat noise as speech).
- **`min_speech_duration_ms` (int, ms)**: detected speech segments shorter than this are discarded (filters out clicks / tiny bursts).
- **`max_speech_duration_s` (float, seconds)**: splits very long speech segments (primarily relevant for segmenting long recordings).
- **`min_silence_duration_ms` (int, ms)**: how long silence must persist at the end of a segment before the segment is closed (helps handle hesitation/short pauses).
- **`window_size_samples` (int, samples @ 16kHz)**: VAD analysis window size. Silero is typically trained for `512`, `1024`, or `1536` at 16kHz; other values can hurt performance.
- **`speech_pad_ms` (int, ms)**: padding added around detected speech segments to reduce overly-tight cuts.

### Where these are applied

`ReplyOnPause` accepts both as parameters:

```python
ReplyOnPause(
    fn=your_reply_fn,
    algo_options=AlgoOptions(...),
    model_options=SileroVadOptions(...),
)
```

In this repo, `voice_chat.py` constructs `ReplyOnPause(response, algo_options=..., model_options=...)` so you can tune it per environment.
