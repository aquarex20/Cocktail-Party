# CocktailPartyAI

A real-time voice assistant system for conversational AI interactions, featuring advanced Voice Activity Detection (VAD), speech-to-text transcription, and text-to-speech synthesis.

## 🎯 Overview

CocktailPartyAI enables real-time voice conversations with AI models. The system listens to your microphone, transcribes speech, sends it to an AI API, and speaks the response back to you—all in real-time with intelligent voice activity detection.

## ⭐ Main Component: `ai_tts.py`

**`Cocktail-Party/Audio_Analysis/ai_tts.py`** is the core working application—a fully functional real-time voice assistant with a graphical interface.

### Features

- **🎤 Voice Activity Detection (VAD)**: Automatically detects when you're speaking using WebRTC VAD with automatic calibration
- **📝 Speech-to-Text**: Transcribes your speech using Faster Whisper (offline, fast, accurate)
- **🤖 AI Integration**: Connects to local AI models via Ollama API for conversational responses
- **🔊 Text-to-Speech**: Speaks AI responses using Coqui TTS
- **🎛️ Calibration System**: Automatically calibrates VAD thresholds based on your environment (ambient noise, speech levels)
- **🎙️ Push-to-Talk Mode**: Alternative recording mode for manual control
- **📊 Real-time VU Meter**: Visual feedback of microphone input levels
- **⚙️ Adjustable VAD Aggressiveness**: Fine-tune voice detection sensitivity

### How It Works

1. **Calibration**: The system measures ambient noise and your speech levels to optimize detection thresholds
2. **Listening**: VAD continuously monitors audio input and detects when you start speaking
3. **Transcription**: Detected speech is transcribed using Whisper
4. **AI Processing**: Transcripts are sent to your local AI model (Ollama) with conversation context
5. **Response**: AI responses are synthesized to speech and played back

## 📋 Prerequisites

- Python 3.8 or higher
- A microphone
- [Ollama](https://ollama.ai/) installed and running locally (for AI responses)
- An Ollama model downloaded (default: `llama3.2:latest`)

## 🚀 Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd CocktailPartyAI
   ```

2. **Navigate to the Audio Analysis directory:**

   ```bash
   cd Cocktail-Party/Audio_Analysis
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes all necessary packages:

   - `numpy` - Numerical operations
   - `scipy` - Signal processing (resampling)
   - `requests` & `httpx` - HTTP client for AI API
   - `sounddevice` - Audio I/O
   - `webrtcvad` - Voice Activity Detection
   - `TTS` - Coqui Text-to-Speech
   - `faster-whisper` - Fast Whisper speech recognition

4. **Set up Ollama (if not already installed):**
   ```bash
   # Install Ollama from https://ollama.ai/
   # Pull a model (e.g., llama3.2:latest)
   ollama pull llama3.2:latest
   ```

## 🎮 Usage

1. **Start Ollama (if not running):**

   ```bash
   ollama serve
   ```

2. **Run the application:**

   ```bash
   python ai_tts.py
   ```

3. **Using the GUI:**
   - **Select Microphone**: Choose your input device from the dropdown
   - **🧭 Calibrate**: Click to calibrate VAD (stay quiet for 1.5s, then speak clearly for 2s)
   - **🎤 Start Listening (VAD)**: Begin automatic voice detection mode
   - **⛔ Stop Listening**: Stop the VAD system
   - **🎙 Push-to-Talk (hold)**: Hold the button to record manually
   - **🔊 Speak Text**: Send the text in the input area to AI and speak the response
   - **🧹 Clear**: Clear the transcript text area

## ⚙️ Configuration

You can modify settings in `ai_tts.py`:

- **AI Model**: Change `AI_MODEL` (line 51) to use a different Ollama model
- **AI API URL**: Modify `AI_API_URL` (line 50) if Ollama runs on a different port
- **STT Model**: Adjust `STT_MODEL` (line 43) - options: `"small.en"`, `"medium.en"`, `"large-v3"`
- **TTS Model**: Change `TTS_MODEL` (line 39) for different voice synthesis
- **Initial Prompt**: Customize `INITIAL_PROMPT` (lines 52-59) to change AI behavior

## 📁 Project Structure

```
CocktailPartyAI/
├── Cocktail-Party/
│   ├── Audio_Analysis/          ⭐ Main working component
│   │   ├── ai_tts.py            ← Core voice assistant application
│   │   └── requirements.txt     ← Dependencies
│   ├── Video_Analysis/          (Experimental video analysis components)
│   └── ...                      (Other audio/video utilities)
└── diart/                       (Speaker diarization library)
```

## 🔧 Troubleshooting

- **"Cannot connect to AI API"**: Make sure Ollama is running (`ollama serve`)
- **No microphone detected**: Check your system audio settings and ensure the mic is not being used by another application
- **Poor VAD detection**: Run calibration again in a quieter environment
- **Slow transcription**: Try using `"small.en"` instead of `"medium.en"` for faster (but less accurate) transcription
- **TTS model download**: The first run will download the TTS model (~100MB), which may take a few minutes

## 📝 Notes

- The application uses **offline processing** for STT (no internet required for transcription)
- AI responses require Ollama to be running locally
- Calibration significantly improves VAD accuracy—always calibrate in your actual usage environment
- The system maintains conversation context by keeping the last few exchanges in memory

## 🎯 Other Components

The repository also contains experimental video analysis tools in `Video_Analysis/` and the `diart` speaker diarization library, but **`ai_tts.py` is the main production-ready component**.

---

**Enjoy your real-time AI conversations! 🎉**
