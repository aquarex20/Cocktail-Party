"""
Push-to-Talk Audio AI Assistant - GUI Version
Click and hold the button to record, release to send to AI and hear response.

FEATURES:
- GUI Button: Hold to record, release to process
- Visual feedback: Shows recording status, transcript, AI response
- Simple and clean interface
"""

import numpy as np
import pyaudio
import time
import json
import requests
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

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
    GTTS_AVAILABLE = True
    # Don't import pygame at module level - causes segfaults
    # Import it lazily when needed
    PYGAME_AVAILABLE = None  # Will check when needed
except ImportError:
    GTTS_AVAILABLE = False
    PYGAME_AVAILABLE = False

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
Do not include stage directions or descriptions of physical actions. Only write text or dialogue.
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
        self.audio_chunks = []
        
        try:
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            raise Exception(f"Failed to initialize PyAudio: {e}")
    
    def start_recording(self):
        """Start recording audio."""
        # CRITICAL: Make sure any existing stream is fully closed first
        if self.stream:
            try:
                if self.stream.is_active():
                    print("stream was actually active so we try stopping it")
                    self.stream.stop_stream()
                    print("stream is stopped")

                self.stream.close()
                print("stream is closed")
            except:
                pass
            self.stream = None
            # Small delay to ensure stream is fully closed
            time.sleep(0.05)
        
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
            print(f"✅ Recording stream started (chunks: {len(self.audio_chunks)})")
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            self.stream = None
            raise Exception(f"Failed to start recording: {e}")
    
    def record_chunk(self):
        """Record one chunk of audio."""
        if not self.stream or not self.stream.is_active():
            print(" record_chunk:stream is not active so we return None")
            return None
        
        try:
            audio_bytes = self.stream.read(self.chunk_size, exception_on_overflow=False)
            print(" record_chunk:audio_bytes is read")
            if not audio_bytes or len(audio_bytes) == 0:
                print(" record_chunk:audio_bytes is not read or is empty so we return None")
                return None
            
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_data = audio_data.astype(np.float32)
            audio_data = audio_data / 32768.0
            
            self.audio_chunks.append(audio_data)
            return audio_data
        except Exception as e:
            print(f"⚠️  Error recording chunk: {e}")
            return None
    
    def stop_recording(self):
        """Stop recording and return all audio chunks."""
        if not self.stream:
            print("⚠️  No stream to stop")
            return np.array([])
        
        try:
            if self.stream.is_active():
                print(" stop_recording:stream is active so we try stopping it")
                self.stream.stop_stream()
                print(" stop_recording:stream is stopped")
            self.stream.close()
            print(" stop_recording:stream is closed")
        except Exception as e:
            print(f"⚠️  Error stopping stream: {e}")
        finally:
            self.stream = None
            # Small delay to ensure stream is fully closed
            time.sleep(0.05)
        
        chunk_count = len(self.audio_chunks)
        print(f"✅ Stopped recording: {chunk_count} chunks collected")
        
        if chunk_count == 0:
            print("⚠️  No audio chunks recorded!")
            return np.array([])
        
        # Concatenate all chunks safely
        try:
            audio_data = np.concatenate(self.audio_chunks)
            print("step 1 ")
            self.audio_chunks = []
            print("step 2 ")

            duration = len(audio_data) / self.sample_rate

            print(f"✅ Audio data: {len(audio_data)} samples = {duration:.2f}s")
            return audio_data
        except Exception as e:
            print(f"⚠️  Error concatenating audio: {e}")
            self.audio_chunks = []
            return np.array([])
    
    def __del__(self):
        """Cleanup."""
        print("🧹 Cleaning up AudioRecorder...")
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass
        print("✅ AudioRecorder cleaned up")


class Transcriber:
    """Transcribes audio to text."""
    
    def __init__(self, method="whisper"):
        self.method = method
        self.whisper_model = None
        self.recognizer = None
        self._model_loaded = False
        
        # Don't load Whisper immediately - load it lazily when needed
        if method == "whisper" and WHISPER_AVAILABLE:
            # Just mark that we'll use Whisper, don't load it yet
            self._use_whisper = True
        else:
            self._use_whisper = False
        
        if method == "google" or (not self._use_whisper and SPEECH_RECOGNITION_AVAILABLE):
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = 300
                print("✅ Google Speech Recognition ready")
            except Exception as e:
                print(f"⚠️  Failed to initialize Google SR: {e}")
    
    def _load_whisper_model(self):
        """Lazy load Whisper model when needed."""
        if self._model_loaded:
            return
        
        if not self._use_whisper or not WHISPER_AVAILABLE:
            return
        
        try:
            print("🔄 Loading Whisper model (this may take a moment)...")
            self.whisper_model = whisper.load_model("base")
            self._model_loaded = True
            print("✅ Whisper model loaded")
        except Exception as e:
            print(f"⚠️  Failed to load Whisper: {e}")
            self.whisper_model = None
            # Fallback to Google if available
            if SPEECH_RECOGNITION_AVAILABLE and not self.recognizer:
                try:
                    self.recognizer = sr.Recognizer()
                    self.recognizer.energy_threshold = 300
                    print("✅ Fallback to Google Speech Recognition")
                except:
                    pass
    
    def transcribe(self, audio_data, sample_rate=16000):
        """Transcribe audio data to text."""
        if len(audio_data) == 0:
            return None
        
        # Lazy load Whisper model when actually needed
        if self._use_whisper and not self._model_loaded:
            self._load_whisper_model()
        
        try:
            if self.whisper_model:
                result = self.whisper_model.transcribe(audio_data, language="en")
                transcript = result["text"].strip()
                if transcript:
                    return transcript
                return None
            
            elif self.recognizer:
                audio_int16 = (audio_data * 32768.0).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                audio_source = sr.AudioData(audio_bytes, sample_rate, 2)
                
                transcript = self.recognizer.recognize_google(audio_source, language="en-US")
                if transcript:
                    return transcript
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
        self._initialized = False
        self.is_speaking = False
        self.speak_lock = threading.Lock()
        self._utterance_finished_event = threading.Event()
        self._on_finished_callback = None
        # Don't initialize TTS immediately - do it lazily when needed
    
    def _initialize(self):
        """Lazy initialize TTS engine when needed."""
        if self._initialized:
            return
        
        self._initialized = True
        
        if self.method == "pyttsx3" or (self.method == "auto" and PYTTSX3_AVAILABLE):
            try:
                self.engine = pyttsx3.init()
                self.use_pyttsx3 = True
                
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
        
        if not self.use_pyttsx3 and (self.method == "gtts" or self.method == "auto"):
            if GTTS_AVAILABLE:
                print("✅ gTTS TTS available")
            else:
                print("⚠️  No TTS available")
    
    def speak(self, text, on_finished=None):
        """Speak text and wait for completion.
        
        Args:
            text: Text to speak
            on_finished: Optional callback function to call when speech finishes
        """
        if not text or len(text.strip()) == 0:
            return
        
        # Lazy initialize TTS when actually needed
        if not self._initialized:
            self._initialize()
        
        with self.speak_lock:
            if self.is_speaking:
                print("⚠️  TTS already speaking, skipping...")
                return
            self.is_speaking = True
        
        text = text.strip()
        
        try:
            if self.use_pyttsx3 and self.engine:
                # Use event callbacks for reliable detection of when speech finishes
                # This is more accurate than runAndWait() alone
                self._utterance_finished_event.clear()
                self._on_finished_callback = on_finished
                
                # Define event handlers
                def on_end(name, completed):
                    """Called when utterance finishes.
                    This runs in pyttsx3's internal thread.
                    Use threading.Event for thread-safe flag setting.
                    """
                    # Set event flag (thread-safe)
                    self._utterance_finished_event.set()
                
                # Connect to the finished-utterance event
                # This fires when speech actually finishes (not just when synthesis completes)
                self.engine.connect('finished-utterance', on_end)
                
                # Speak the text
                self.engine.say(text)
                # runAndWait() waits for synthesis, but the event callback
                # will fire when audio actually finishes playing
                self.engine.runAndWait()
                
                # Wait for the event to fire (should happen during runAndWait, but double-check)
                # This ensures we wait until the actual audio playback finishes
                # Use timeout to avoid infinite wait (should fire almost immediately)
                self._utterance_finished_event.wait(timeout=5.0)
                
                # Small buffer for audio system to fully finish
                time.sleep(0.2)
                
                # Call callback from this thread context (the callback itself uses root.after() for GUI updates)
                if self._on_finished_callback:
                    try:
                        self._on_finished_callback()
                    except Exception as e:
                        print(f"⚠️  Error in on_finished callback: {e}")
            
            elif GTTS_AVAILABLE:
                # Lazy import pygame to avoid segfaults
                try:
                    import pygame
                    PYGAME_AVAILABLE = True
                except ImportError:
                    print("⚠️  pygame not available for gTTS")
                    return
                
                try:
                    tts = gTTS(text=text, lang='en', slow=False)
                    tts.save('/tmp/tts_output.mp3')
                    pygame.mixer.init()
                    pygame.mixer.music.load('/tmp/tts_output.mp3')
                    pygame.mixer.music.play()
                    # According to pygame documentation:
                    # - get_busy() returns True while a sound is playing
                    # - Returns False when playback has stopped
                    # Check until audio actually finishes playing
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)  # Check frequently for accuracy
                    
                    # Extra buffer time for audio system to finish
                    # (some systems need time for audio buffers to drain)
                    time.sleep(0.2)
                    
                    # Mark as finished and call callback when gTTS audio finishes
                    self._utterance_finished_event.set()
                    if on_finished:
                        try:
                            on_finished()
                        except Exception as e:
                            print(f"⚠️  Error in on_finished callback: {e}")
                finally:
                    # CRITICAL: Always cleanup pygame
                    try:
                        pygame.mixer.stop()
                        pygame.mixer.quit()
                    except:
                        pass
        except Exception as e:
            print(f"⚠️  TTS error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self.speak_lock:
                self.is_speaking = False
                # Call callback when speech finishes (only if not already called)
                # For pyttsx3, callback is called after event fires
                # For gTTS, callback is called after audio finishes
                # This is only for error cases or if callback wasn't called
                if on_finished and not self._utterance_finished_event.is_set():
                    try:
                        on_finished()
                    except Exception as e:
                        print(f"⚠️  Error in on_finished callback: {e}")
                # Reset flags
                self._utterance_finished_event.clear()
                self._on_finished_callback = None
    
    def is_currently_speaking(self):
        """Check if currently speaking."""
        with self.speak_lock:
            return self.is_speaking
    
    def wait_until_done(self):
        """Wait until speech finishes."""
        while self.is_speaking:
            time.sleep(0.1)


def call_ai(transcript):
    """Call AI API and return response."""
    if not transcript:
        return None
    
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
            return full_response.strip()
        else:
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to AI API at {AI_API_URL}")
        return None
    except requests.exceptions.Timeout:
        print(f"❌ AI API timeout (>30s)")
        return None
    except Exception as e:
        print(f"❌ AI API error: {e}")
        return None


class PushToTalkGUI:
    """GUI for push-to-talk audio assistant."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Push-to-Talk AI Assistant")
        self.root.geometry("600x500")
        
        self.recording = False
        self.processing = False
        self.recording_start_time = None  # Track when recording started
        self.processing_thread = None  # Track processing thread
        
        # Initialize components
        try:
            self.recorder = AudioRecorder(sample_rate=16000, chunk_size=1024)
        except Exception as e:
            self.show_error(f"Failed to initialize audio: {e}")
            return
        
        try:
            self.transcriber = Transcriber(method="google")  # Use Google instead of Whisper to avoid segfault
            if not self.transcriber.recognizer:
                # Fallback to Whisper if Google not available
                print("⚠️  Google not available, trying Whisper...")
                self.transcriber = Transcriber(method="whisper")
        except Exception as e:
            self.show_error(f"Failed to initialize transcriber: {e}")
            return
        
        try:
            # TEMPORARY: Disable TTS to isolate segfault issue
            # Set to False to disable TTS completely
            ENABLE_TTS = True  # Set to False to disable TTS
            
            if ENABLE_TTS:
                # Don't initialize TTS immediately - it causes segfaults
                # Will initialize lazily when first needed
                # Create minimal TTS object - no initialization
                self.tts = TextToSpeech(method="auto")
                print("✅ TTS ready (will initialize when needed)")
            else:
                self.tts = None
                print("⚠️  TTS disabled (to avoid segfaults)")
        except Exception as e:
            print(f"⚠️  TTS initialization warning: {e}")
            self.tts = None
        
        # Setup UI last - after all components are initialized
        print("🔄 Setting up GUI...")
        try:
            self.setup_ui()
            print("✅ GUI ready")
        except Exception as e:
            self.show_error(f"Failed to setup GUI: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            self.start_recording_loop()
            print("✅ Recording loop started")
        except Exception as e:
            print(f"⚠️  Recording loop warning: {e}")
    
    def setup_ui(self):
        """Setup the GUI."""
        # Title
        title = tk.Label(
            self.root,
            text="Push-to-Talk AI Assistant",
            font=("Arial", 18, "bold"),
            pady=10
        )
        title.pack()
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready - Hold button to record",
            font=("Arial", 12),
            fg="green",
            pady=5
        )
        self.status_label.pack()
        
        # Record button
        self.record_button = tk.Button(
            self.root,
            text="🎤 HOLD TO RECORD",
            font=("Arial", 16, "bold"),
            bg="#4CAF50",
            fg="white",
            relief="raised",
            padx=20,
            pady=15,
            cursor="hand2"
        )
        self.record_button.pack(pady=20)
        
        # Bind button events
        self.record_button.bind("<ButtonPress-1>", self.on_button_press)
        self.record_button.bind("<ButtonRelease-1>", self.on_button_release)
        
        # Transcript display
        transcript_label = tk.Label(
            self.root,
            text="Transcript:",
            font=("Arial", 10, "bold"),
            anchor="w"
        )
        transcript_label.pack(fill="x", padx=20, pady=(10, 5))
        
        self.transcript_text = scrolledtext.ScrolledText(
            self.root,
            height=5,
            font=("Arial", 10),
            wrap=tk.WORD
        )
        self.transcript_text.pack(fill="both", expand=True, padx=20, pady=(0, 10))
    
    def show_error(self, message):
        """Show error message."""
        error_label = tk.Label(
            self.root,
            text=f"❌ {message}",
            font=("Arial", 12),
            fg="red",
            wraplength=550
        )
        error_label.pack(pady=20)
    
    def update_status(self, message, color="black"):
        """Update status label."""
        self.status_label.config(text=message, fg=color)
        self.root.update()
    
    def append_text(self, text):
        """Append text to transcript display."""
        self.transcript_text.insert(tk.END, text + "\n")
        self.transcript_text.see(tk.END)
        self.root.update()
    
    def on_button_press(self, event):
        """Handle button press (start recording)."""
        # CRITICAL: Don't allow recording if processing or TTS is speaking
        # Also check if previous thread is still alive
        if self.processing or (self.processing_thread and self.processing_thread.is_alive()):
            if self.processing_thread and self.processing_thread.is_alive():
                self.update_status("⏳ Previous processing still running...", "orange")
                print("⚠️  Previous processing thread still running, ignoring press")
            else:
                self.update_status("⏳ Still processing previous request...", "orange")
            return
        
        if self.tts and self.tts.is_currently_speaking():
            print("tts is still speaking peasant!")
            self.update_status("🔊 Wait for speech to finish...", "blue")
            return
        
        if not self.recording:
            print("recording is not recording peasant!")
            self.recording = True
            self.recording_start_time = time.time()  # Record start time
            try:
                # Clear any old audio data first
                self.recorder.audio_chunks = []
                self.recorder.start_recording()
                self.update_status("🔴 Recording... Hold button!", "red")
                self.record_button.config(bg="#f44336", text="🔴 RECORDING...")
            except Exception as e:
                self.update_status(f"Error: {e}", "red")
                self.recording = False
                self.recording_start_time = None
                print(f"❌ Error starting recording: {e}")
                import traceback
                traceback.print_exc()
    
    def on_button_release(self, event):
        """Handle button release (stop recording and process)."""
        if not self.recording or self.processing:
            return
        
        self.recording = False
        
        # Calculate actual recording duration
        actual_duration = 0
        if self.recording_start_time:
            actual_duration = time.time() - self.recording_start_time
            self.recording_start_time = None
        
        # Stop recording
        audio_data = self.recorder.stop_recording()
        
        # Debug: Show what we got
        print(f"🔍 Debug: Got {len(audio_data)} samples after stop_recording()")
        
        # Show recording duration
        if actual_duration > 0:
            self.update_status(f"⏳ Processed {actual_duration:.2f}s recording...", "orange")
        else:
            self.update_status("⏳ Processing...", "orange")
        print("step 3 ")
        self.record_button.config(bg="#9E9E9E", text="⏳ PROCESSING...", state="disabled")
        print("step 4 ")
        # Process in background thread (daemon=True so app can exit if needed)
        # Store reference so we can check if it's still running
        self.processing_thread = threading.Thread(target=self.process_audio, args=(audio_data,), daemon=True)
        print("step 5 ")
        self.processing_thread.start()
        #self.process_audio(audio_data)
        print("step 6 ")
    
    def process_audio(self, audio_data):
        """Process audio in background thread."""
        self.processing = True
        
        try:
            duration = len(audio_data) / 16000
            print("step 7 ")
            self.root.after(0, lambda: self.append_text(f"📊 Audio duration: {duration:.2f}s ({len(audio_data)} samples)"))
            print("step 8 ")
            # Lower threshold - 0.2 seconds minimum (very short, but catches empty recordings)
            if duration < 0.2:
                self.root.after(0, lambda: self.update_status(f"⚠️  Recording too short ({duration:.2f}s)", "orange"))
                self.root.after(0, lambda: self.append_text(f"(Recording too short - need at least 0.2s, got {duration:.2f}s. Hold button longer!)"))
                self.processing = False
                self.root.after(0, self.reset_button)
                return
            print("step 9 ")
            # Transcribe (Whisper will load lazily here if needed)
            self.root.after(0, lambda: self.update_status("🎤 Transcribing...", "orange"))
            transcript = self.transcriber.transcribe(audio_data, sample_rate=16000)
            print("step 10 ")
            if not transcript:
                self.root.after(0, lambda: self.update_status("⚠️  No transcript", "orange"))
                self.root.after(0, lambda: self.append_text("(Could not transcribe audio)"))
                self.processing = False
                self.root.after(0, self.reset_button)
                return
            print("step 11 ")
            # Show transcript
            self.root.after(0, lambda: self.append_text(f"📝 You said: {transcript}"))
            print("step 12 ")

            # Call AI
            self.root.after(0, lambda: self.update_status("🤖 Calling AI...", "orange"))
            ai_response = call_ai(transcript)
            print("step 13 ")

            if not ai_response:
                self.root.after(0, lambda: self.update_status("⚠️  No AI response", "orange"))
                self.root.after(0, lambda: self.append_text("(No response from AI)"))
                self.processing = False
                self.root.after(0, self.reset_button)
                return
            print("step 14 ")

            # Show AI response
            self.root.after(0, lambda: self.append_text(f"🤖 AI: {ai_response}"))
            print("step 15 ")
            # Speak response (if TTS enabled)
            if self.tts:
                self.root.after(0, lambda: self.update_status("🔊 Speaking response...", "blue"))
                try:
                    # Define callback to show when speech finishes
                    def on_speech_finished():
                        self.root.after(0, lambda: self.update_status("✅ Speech finished! Ready for next recording", "green"))
                    
                    self.tts.speak(ai_response, on_finished=on_speech_finished)
                    # Wait for TTS to finish completely
                    self.tts.wait_until_done()
                    print("✅ TTS finished")
                except Exception as e:
                    print(f"⚠️  TTS error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # TTS disabled - just show response
                print("ℹ️  TTS disabled - response shown in text only")
                self.root.after(0, lambda: self.update_status("✅ Complete! Hold button to record again", "green"))
            
        except Exception as e:
            print(f"❌ Process audio error: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.update_status(f"Error: {e}", "red"))
            self.root.after(0, lambda: self.append_text(f"(Error: {e})"))
        finally:
            # CRITICAL: Always reset processing flag and button
            self.processing = False
            self.processing_thread = None  # Clear thread reference
            self.root.after(0, self.reset_button)
            print("✅ Processing complete, ready for next recording")
    
    def reset_button(self):
        """Reset button to ready state."""
        self.record_button.config(
            bg="#4CAF50",
            text="🎤 HOLD TO RECORD",
            state="normal"
        )
    
    def start_recording_loop(self):
        """Start loop to record chunks while button is held."""
        def record_loop():
            chunk_count = 0
            while True:
                if self.recording:
                    try:
                        chunk = self.recorder.record_chunk()
                        if chunk is not None:
                            chunk_count += 1
                            # Debug: print every 50 chunks
                            if chunk_count % 50 == 0:
                                print(f"📊 Recorded {chunk_count} chunks ({chunk_count * 1024 / 16000:.2f}s)...")
                        elif self.recorder.stream and not self.recorder.stream.is_active():
                            # Stream died unexpectedly
                            print("⚠️  Recording stream died unexpectedly")
                            self.recording = False
                            self.root.after(0, self.reset_button)
                    except Exception as e:
                        print(f"⚠️  Error recording chunk: {e}")
                        import traceback
                        traceback.print_exc()
                        self.recording = False
                        self.root.after(0, self.reset_button)
                else:
                    chunk_count = 0  # Reset when not recording
                time.sleep(0.01)  # Small delay
        
        threading.Thread(target=record_loop, daemon=True).start()


def main():
    """Main function."""
    if not TRANSCRIPTION_AVAILABLE:
        print("❌ No transcription method available.")
        print("   Install: pip install SpeechRecognition")
        print("   Or: pip install openai-whisper")
        return
    
    print("🔄 Creating GUI window...")
    try:
        root = tk.Tk()
        print("✅ GUI window created")
    except Exception as e:
        print(f"❌ Failed to create GUI window: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("🔄 Initializing application...")
    try:
        app = PushToTalkGUI(root)
        print("✅ Application initialized")
    except Exception as e:
        print(f"❌ Failed to initialize application: {e}")
        import traceback
        traceback.print_exc()
        root.destroy()
        return
    
    print("✅ Starting GUI main loop...")
    try:
        root.mainloop()
    except Exception as e:
        print(f"❌ GUI error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

