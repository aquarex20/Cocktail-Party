"""
Push-to-Talk Audio AI Assistant
Hold BACKSPACE to record, release to send to AI and hear response.

FEATURES:
- Hold BACKSPACE: Records microphone
- Release BACKSPACE: Transcribes, sends to AI, reads response
- Simple, one action at a time
"""

import numpy as np
import pyaudio
import time
import json
import requests
import threading
from collections import deque

# Don't import keyboard at module level - it causes Bus errors on macOS
# We'll import it lazily in main() instead
KEYBOARD_AVAILABLE = None  # Will be set when we try to import

# Try to import text-to-speech
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("⚠️  pyttsx3 not installed - TTS unavailable")
    print("   Install with: pip install pyttsx3")

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Try to import speech recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️  speech_recognition not installed - transcription unavailable")
    print("   Install with: pip install SpeechRecognition")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  OpenAI Whisper not installed - offline transcription unavailable")
    print("   Install with: pip install openai-whisper")

TRANSCRIPTION_AVAILABLE = SPEECH_RECOGNITION_AVAILABLE or WHISPER_AVAILABLE

# AI Configuration
AI_API_URL = "http://localhost:11434/api/generate"
AI_MODEL = "llama3.2:latest"  # Change to your model name
INITIAL_PROMPT = """
You are a helpful assistant that can answer questions and help with tasks.
You are currently in a cocktail party.
You are sitting at a table with a group of people.
You are listening to the conversation and trying to understand what is going on.
You are also trying to say something interesting to the group or show interest in the conversation.
Here's what the group is talking about:
"""


class AudioRecorder:
    """Records audio chunks."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = None
        self.stream = None
        self.audio_chunks = []  # Store recorded chunks
        
        # Initialize PyAudio with error handling
        try:
            self.audio = pyaudio.PyAudio()
            print(f"✅ Audio initialized: {self.sample_rate}Hz, chunk_size={self.chunk_size}")
        except Exception as e:
            print(f"❌ Failed to initialize PyAudio: {e}")
            raise
    
    def start_recording(self):
        """Start recording audio."""
        if self.stream:
            return
        
        if not self.audio:
            print("❌ Audio not initialized")
            return
        
        self.audio_chunks = []  # Clear previous recording
        
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.stream.start_stream()
            print("🎤 Recording...")
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            self.stream = None
            raise
    
    def record_chunk(self):
        """Record one chunk of audio (non-blocking)."""
        if not self.stream or not self.stream.is_active():
            return None
        
        try:
            # Read chunk (non-blocking)
            audio_bytes = self.stream.read(self.chunk_size, exception_on_overflow=False)
            
            if not audio_bytes or len(audio_bytes) == 0:
                return None
            
            # Convert to numpy array - use safer method
            try:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_data = audio_data.astype(np.float32)
                # Normalize to [-1, 1]
                audio_data = audio_data / 32768.0
            except Exception as e:
                print(f"⚠️  Error converting audio: {e}")
                return None
            
            # Store chunk
            self.audio_chunks.append(audio_data)
            return audio_data
        except Exception as e:
            print(f"⚠️  Error recording chunk: {e}")
            return None
    
    def stop_recording(self):
        """Stop recording and return all audio chunks."""
        if not self.stream:
            return np.array([])
        
        try:
            self.stream.stop_stream()
            self.stream.close()
        except:
            pass
        finally:
            self.stream = None
        
        if len(self.audio_chunks) == 0:
            return np.array([])
        
        # Concatenate all chunks safely
        try:
            audio_data = np.concatenate(self.audio_chunks)
            self.audio_chunks = []
            
            print(f"✅ Stopped recording ({len(audio_data) / self.sample_rate:.2f}s)")
            return audio_data
        except Exception as e:
            print(f"⚠️  Error concatenating audio: {e}")
            self.audio_chunks = []
            return np.array([])
    
    def __del__(self):
        """Cleanup."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass


class Transcriber:
    """Transcribes audio to text."""
    
    def __init__(self, method="whisper"):
        self.method = method
        self.whisper_model = None
        self.recognizer = None
        
        if method == "whisper" and WHISPER_AVAILABLE:
            try:
                print("🔄 Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
                print("✅ Whisper model loaded")
            except Exception as e:
                print(f"⚠️  Failed to load Whisper: {e}")
                self.whisper_model = None
        
        if method == "google" or (method == "auto" and not self.whisper_model and SPEECH_RECOGNITION_AVAILABLE):
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = 300
                print("✅ Google Speech Recognition ready")
            except Exception as e:
                print(f"⚠️  Failed to initialize Google SR: {e}")
    
    def transcribe(self, audio_data, sample_rate=16000):
        """Transcribe audio data to text."""
        if len(audio_data) == 0:
            return None
        
        try:
            if self.whisper_model:
                print("🎤 Transcribing with Whisper...")
                result = self.whisper_model.transcribe(audio_data, language="en")
                transcript = result["text"].strip()
                if transcript:
                    print(f"✅ Transcript: {transcript}")
                    return transcript
                else:
                    print("⚠️  No transcript from Whisper")
                    return None
            
            elif self.recognizer:
                print("🎤 Transcribing with Google...")
                # Convert to AudioData format
                audio_int16 = (audio_data * 32768.0).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                audio_source = sr.AudioData(audio_bytes, sample_rate, 2)
                
                transcript = self.recognizer.recognize_google(audio_source, language="en-US")
                if transcript:
                    print(f"✅ Transcript: {transcript}")
                    return transcript
                else:
                    print("⚠️  No transcript from Google")
                    return None
        except Exception as e:
            print(f"⚠️  Transcription error: {e}")
            return None
        
        return None


class TextToSpeech:
    """Reads text out loud."""
    
    def __init__(self, method="auto"):
        self.method = method
        self.engine = None
        self.use_pyttsx3 = False
        
        if method == "pyttsx3" or (method == "auto" and PYTTSX3_AVAILABLE):
            try:
                self.engine = pyttsx3.init()
                self.use_pyttsx3 = True
                
                # Configure voice
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if 'en' in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.9)
                print("✅ pyttsx3 TTS initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize pyttsx3: {e}")
                self.use_pyttsx3 = False
        
        if not self.use_pyttsx3 and (method == "gtts" or method == "auto"):
            if GTTS_AVAILABLE:
                print("✅ gTTS TTS available")
    
    def speak(self, text):
        """Speak text and wait for completion."""
        if not text or len(text.strip()) == 0:
            return
        
        text = text.strip()
        print(f"🔊 Speaking: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        try:
            if self.use_pyttsx3 and self.engine:
                self.engine.say(text)
                self.engine.runAndWait()
                # Safety delay
                time.sleep(0.3)
                print("✅ Finished speaking")
            
            elif GTTS_AVAILABLE:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save('/tmp/tts_output.mp3')
                pygame.mixer.init()
                pygame.mixer.music.load('/tmp/tts_output.mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                time.sleep(0.3)  # Safety delay
                pygame.mixer.quit()
                print("✅ Finished speaking")
            else:
                print(f"⚠️  TTS not available")
        except Exception as e:
            print(f"⚠️  TTS error: {e}")


def call_ai(transcript):
    """Call AI API and return response."""
    if not transcript:
        return None
    
    print(f"🤖 Calling AI API: {AI_MODEL}...")
    
    try:
        response = requests.post(
            AI_API_URL,
            json={
                "model": AI_MODEL,
                "prompt": INITIAL_PROMPT + transcript,
                "stream": True
            },
            stream=True,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        response.raise_for_status()
        
        # Collect full response
        print("   Collecting complete response...")
        full_response = ""
        response_complete = False
        
        while not response_complete:
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    if "response" in data:
                        full_response += data["response"]
                    
                    if data.get("done", False):
                        response_complete = True
                        break
                    
                    if "error" in data:
                        print(f"⚠️  AI API error: {data['error']}")
                        return None
                except json.JSONDecodeError:
                    continue
            
            if not response_complete:
                if full_response.strip():
                    response_complete = True
                else:
                    break
        
        if full_response.strip():
            print(f"✅ AI response complete:")
            print(full_response)
            return full_response.strip()
        else:
            print("⚠️  No response from AI")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to AI API at {AI_API_URL}")
        print("   Make sure Ollama is running: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print(f"❌ AI API timeout (>30s)")
        return None
    except Exception as e:
        print(f"❌ AI API error: {e}")
        return None


def main():
    """Main function - push to talk."""
    print("=" * 60)
    print("Push-to-Talk Audio AI Assistant")
    print("=" * 60)
    
    # Try to import keyboard (lazy import to avoid Bus errors)
    print("\n0️⃣ Checking keyboard library...")
    try:
        import keyboard
        print("✅ Keyboard library available")
    except ImportError:
        print("❌ Keyboard library not installed")
        print("   Install: pip install keyboard")
        return
    except Exception as e:
        print(f"❌ Keyboard library error: {e}")
        print("   On macOS, you may need to:")
        print("   1. Grant accessibility permissions:")
        print("      System Settings > Privacy & Security > Accessibility")
        print("   2. Add Terminal (or your IDE) to allowed apps")
        print("   3. Restart Terminal")
        return
    
    if not TRANSCRIPTION_AVAILABLE:
        print("❌ No transcription method available.")
        print("   Install: pip install SpeechRecognition")
        print("   Or: pip install openai-whisper")
        return
    
    print("Hold BACKSPACE to record, release to process")
    print("=" * 60)
    
    # Initialize components with error handling - one at a time to isolate crashes
    print("\n1️⃣ Initializing audio recorder...")
    try:
        recorder = AudioRecorder(sample_rate=16000, chunk_size=1024)
        print("✅ Audio recorder ready")
    except Exception as e:
        print(f"❌ Failed to initialize audio recorder: {e}")
        print("   Make sure microphone permissions are granted")
        import traceback
        traceback.print_exc()
        return
    
    print("\n2️⃣ Initializing transcriber...")
    try:
        # Try Whisper first, fallback to Google if it crashes
        transcriber = None
        try:
            transcriber = Transcriber(method="whisper")
            if not transcriber.whisper_model and not transcriber.recognizer:
                print("⚠️  Whisper failed, trying Google...")
                transcriber = Transcriber(method="google")
        except Exception as e:
            print(f"⚠️  Whisper initialization failed: {e}")
            print("   Trying Google Speech Recognition...")
            try:
                transcriber = Transcriber(method="google")
            except Exception as e2:
                print(f"❌ Google SR also failed: {e2}")
                return
        
        if not transcriber or (not transcriber.whisper_model and not transcriber.recognizer):
            print("❌ No transcription method available")
            return
        print("✅ Transcriber ready")
    except Exception as e:
        print(f"❌ Failed to initialize transcriber: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n3️⃣ Initializing TTS...")
    try:
        tts = TextToSpeech(method="auto")
        print("✅ TTS ready")
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize TTS: {e}")
        tts = None
    
    print("\n✅ Ready! Hold BACKSPACE to record...")
    print("=" * 60)
    
    try:
        recording = False
        processing = False  # Prevent multiple simultaneous processing
        
        def on_press(event):
            nonlocal recording
            if event.name == 'backspace' and not recording and not processing:
                recording = True
                recorder.start_recording()
                print("🎤 Recording (hold BACKSPACE)...")
        
        def on_release(event):
            nonlocal recording, processing
            if event.name == 'backspace' and recording:
                recording = False
                # Stop recording
                audio_data = recorder.stop_recording()
                
                # Process in separate thread to avoid blocking key detection
                def process_audio():
                    nonlocal processing
                    processing = True
                    
                    try:
                        if len(audio_data) == 0:
                            print("⚠️  No audio recorded")
                            processing = False
                            return
                        
                        # Transcribe
                        transcript = transcriber.transcribe(audio_data, sample_rate=16000)
                        
                        if not transcript:
                            print("⚠️  No transcript, skipping AI call")
                            processing = False
                            return
                        
                        # Call AI
                        ai_response = call_ai(transcript)
                        
                        if not ai_response:
                            print("⚠️  No AI response")
                            processing = False
                            return
                        
                        # Speak response
                        if tts:
                            tts.speak(ai_response)
                        else:
                            print("⚠️  TTS not available, response not spoken")
                        
                        print("\n✅ Complete! Press BACKSPACE again to record...\n")
                        print("=" * 60)
                    finally:
                        processing = False
                
                # Process in background thread
                threading.Thread(target=process_audio, daemon=False).start()
        
        # Hook key events
        try:
            print("   Hooking keyboard events...")
            keyboard.on_press(on_press)
            keyboard.on_release(on_release)
            print("✅ Listening for BACKSPACE key...")
            print("Hold BACKSPACE to record, release to process")
        except Exception as e:
            print(f"❌ Failed to hook keyboard events: {e}")
            print("   On macOS, you may need to grant accessibility permissions")
            print("   System Settings > Privacy & Security > Accessibility")
            print("   Then restart Terminal and try again")
            return
        
        # Keep running - record chunks while key is held
        try:
            while True:
                if recording:
                    try:
                        recorder.record_chunk()
                    except Exception as e:
                        print(f"⚠️  Error recording: {e}")
                        recording = False
                time.sleep(0.01)  # Small delay
        except KeyboardInterrupt:
            pass
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()

