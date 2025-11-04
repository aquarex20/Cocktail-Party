# Transcription Configuration Guide

## Currently Used Services

The script supports two transcription services:

### 1. **OpenAI Whisper** (Recommended)
- **Offline**: Works without internet
- **Better accuracy**: Generally more accurate than Google
- **Slower**: Takes longer to process
- **Install**: `pip install openai-whisper`

### 2. **Google Speech Recognition**
- **Online**: Requires internet connection
- **Faster**: Quick API responses
- **Less accurate**: More "speech not recognized" errors
- **Install**: `pip install SpeechRecognition`

## How to Change the Transcription Service

### Option 1: Edit the Code Directly

Open `video_audio_threaded.py` and find this line (around line 877):

```python
TRANSCRIPTION_METHOD = "whisper"  # Change this to "google", "whisper", or "auto"
```

**Options:**
- `"whisper"` - Use OpenAI Whisper (offline, better accuracy)
- `"google"` - Use Google Speech Recognition (online, faster)
- `"auto"` - Try Whisper first, fallback to Google if not available

### Option 2: Whisper Model Size

If using Whisper, you can change the model size (around line 366):

```python
WHISPER_MODEL_SIZE = "base"  # Options: "tiny", "base", "small", "medium", "large"
```

**Model Sizes:**
- `"tiny"` - Fastest, least accurate (~39MB)
- `"base"` - Good balance (~74MB) - **Recommended**
- `"small"` - Better accuracy, slower (~244MB)
- `"medium"` - Very good accuracy, slow (~769MB)
- `"large"` - Best accuracy, very slow (~1550MB)

### Option 3: Google Settings

If using Google, adjust recognition settings (around line 375):

```python
self.recognizer.energy_threshold = 300  # Lower = more sensitive (try 200-400)
self.recognizer.dynamic_energy_threshold = True
self.recognizer.pause_threshold = 0.8  # Pause detection (0.5-1.5)
```

## Troubleshooting "Speech Not Recognized"

### If using Google Speech Recognition:

1. **Check internet connection** - Google requires internet
2. **Lower energy threshold** - Make it more sensitive:
   ```python
   self.recognizer.energy_threshold = 200
   ```
3. **Speak louder/clearer** - Better audio quality helps
4. **Check microphone** - Make sure mic is working properly
5. **Switch to Whisper** - Whisper is more forgiving with audio quality

### If using Whisper:

1. **Use larger model** - Try "small" or "medium" instead of "base"
2. **Check audio quality** - Make sure audio isn't too noisy
3. **Speak clearly** - Whisper works better with clear speech

## Recommendations

- **For best accuracy**: Use Whisper with "base" or "small" model
- **For fastest results**: Use Google (if internet is stable)
- **For offline use**: Must use Whisper
- **For noisy environments**: Use Whisper (better at handling noise)

## Current Default Settings

- **Method**: `"whisper"` (OpenAI Whisper)
- **Whisper Model**: `"base"` (good balance)
- **Language**: `"en-US"` (English - US)

To change these, edit the code at line 874-877 in `video_audio_threaded.py`


