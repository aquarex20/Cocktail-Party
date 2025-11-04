"""
Audio-Only Analysis with Speech-to-Text Transcription and AI Responses
Simplified version focusing on audio recording, transcription, AI responses, and TTS.

FEATURES:
- ✅ Audio recording: Records microphone input in real-time
- ✅ Transcription: Real-time speech-to-text conversion (English)
- ✅ AI Responses: Sends transcripts to AI and gets responses
- ✅ Text-to-Speech: Reads AI responses out loud
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
import time
import threading
from queue import Queue
import wave
import pyaudio
import json
import requests

# Try to import text-to-speech libraries
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
    print("✅ pyttsx3 available - offline text-to-speech supported")
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("⚠️  pyttsx3 not installed - offline TTS unavailable")
    print("   For offline TTS: pip install pyttsx3")

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
    print("✅ gTTS available - online text-to-speech supported")
except ImportError:
    GTTS_AVAILABLE = False
    print("⚠️  gTTS/pygame not installed - online TTS unavailable")
    print("   For online TTS: pip install gtts pygame")

# AI Prompt
INITIAL_PROMPT = """
You are a helpful assistant that can answer questions and help with tasks.
You are currently in a cocktail party.
You are sitting at a table with a group of people.
You are listening to the conversation and trying to understand what is going on.
You are also trying to say something interesting to the group or show interest in the conversation.
Here's what the group is talking about:
"""

# Try to import speech recognition libraries
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    print("✅ speech_recognition library available - using Google Speech API")
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️  speech_recognition not installed - transcription unavailable")
    print("   For transcription: pip install SpeechRecognition")

try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ OpenAI Whisper available - offline transcription supported")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  OpenAI Whisper not installed - offline transcription unavailable")
    print("   For offline transcription: pip install openai-whisper")

# Check if at least one transcription method is available
TRANSCRIPTION_AVAILABLE = SPEECH_RECOGNITION_AVAILABLE or WHISPER_AVAILABLE


@dataclass
class AudioChunk:
    """Represents an audio chunk from the microphone."""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    is_speaking: bool = False  # Voice activity detection result
    volume: float = 0.0  # RMS volume
    transcript: Optional[str] = None  # Transcribed text (if available)


class TextToSpeech:
    """Handles text-to-speech conversion with good AI voice."""
    
    def __init__(self, method="auto", voice_id=None):
        """
        Initialize text-to-speech.
        
        Args:
            method: "pyttsx3" (offline), "gtts" (online), or "auto" (try pyttsx3 first)
            voice_id: Specific voice ID for pyttsx3 (None for default)
        """
        self.method = method
        self.engine = None
        self.use_pyttsx3 = False
        self.is_speaking = False  # Track if currently speaking
        self.speak_lock = threading.Lock()  # Lock to prevent concurrent speech
        self.speak_queue = Queue()  # Queue for speech requests
        self.speak_thread = None  # Current speaking thread
        
        # Initialize TTS engine
        if method == "pyttsx3" or (method == "auto" and PYTTSX3_AVAILABLE):
            if PYTTSX3_AVAILABLE:
                try:
                    self.engine = pyttsx3.init()
                    self.use_pyttsx3 = True
                    
                    # Configure voice settings for better quality
                    voices = self.engine.getProperty('voices')
                    
                    # Try to find a good female or male voice
                    if voice_id is None:
                        # Prefer English voices
                        for voice in voices:
                            if 'en' in voice.id.lower() or 'english' in voice.name.lower():
                                self.engine.setProperty('voice', voice.id)
                                print(f"✅ Using voice: {voice.name}")
                                break
                    
                    # Set speech rate (words per minute) - default is 200
                    self.engine.setProperty('rate', 150)  # Slower = clearer
                    
                    # Set volume (0.0 to 1.0)
                    self.engine.setProperty('volume', 0.9)  # High volume
                    
                    print("✅ pyttsx3 TTS initialized")
                except Exception as e:
                    print(f"⚠️  Failed to initialize pyttsx3: {e}")
                    self.use_pyttsx3 = False
        
        if not self.use_pyttsx3 and (method == "gtts" or method == "auto"):
            if GTTS_AVAILABLE:
                print("✅ gTTS TTS available (requires internet)")
            else:
                print("⚠️  gTTS not available")
    
    def _speak_internal(self, text: str):
        """Internal method to actually speak the text - ensures completion."""
        if not text or len(text.strip()) == 0:
            return
        
        text = text.strip()
        print(f"🔊 Speaking: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        try:
            if self.use_pyttsx3 and self.engine:
                # Use pyttsx3 - block until completion to prevent interruption
                self.engine.say(text)
                self.engine.runAndWait()  # Block until speech completes
                # Add small delay after runAndWait() to ensure audio actually finishes
                time.sleep(0.3)  # 300ms buffer for audio completion
                print("✅ Finished speaking")
            
            elif GTTS_AVAILABLE:
                # Use gTTS - wait for audio to finish playing
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save('/tmp/tts_output.mp3')
                pygame.mixer.init()
                pygame.mixer.music.load('/tmp/tts_output.mp3')
                pygame.mixer.music.play()
                # Wait for audio to finish - DO NOT INTERRUPT
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                # Add small delay after get_busy() is False to ensure audio actually finishes
                time.sleep(0.3)  # 300ms buffer for audio completion
                pygame.mixer.quit()
                print("✅ Finished speaking")
            
            else:
                print(f"⚠️  TTS not available - cannot speak: {text[:50]}...")
        except Exception as e:
            print(f"⚠️  TTS error: {e}")
        finally:
            # Mark as no longer speaking
            with self.speak_lock:
                self.is_speaking = False
                self.speak_thread = None
            
            # Process next item in queue if any
            self._process_queue()
    
    def _process_queue(self):
        """Process the next item in the speech queue."""
        try:
            if not self.speak_queue.empty():
                next_text = self.speak_queue.get_nowait()
                self.speak(next_text, wait_if_busy=False)
        except:
            pass
    
    def speak(self, text: str, block: bool = False, wait_if_busy: bool = True):
        """
        Speak the given text. If already speaking, queues the request.
        Speech will NOT be interrupted - it will always finish.
        
        Args:
            text: Text to speak
            block: If True, blocks until speech completes. If False, returns immediately.
            wait_if_busy: If True and currently speaking, queue the request. If False and busy, skip.
        """
        if not text or len(text.strip()) == 0:
            return
        
        text = text.strip()
        
        with self.speak_lock:
            if self.is_speaking:
                if wait_if_busy:
                    # Queue the request - will speak after current speech finishes
                    print(f"⏳ Already speaking, queuing: {text[:50]}...")
                    self.speak_queue.put(text)
                else:
                    print(f"⚠️  Already speaking, skipping: {text[:50]}...")
                return
            
            # Mark as speaking
            self.is_speaking = True
        
        # Speak in background thread - will not be interrupted
        def speak_thread():
            self._speak_internal(text)
        
        self.speak_thread = threading.Thread(target=speak_thread, daemon=False)  # Non-daemon to prevent interruption
        self.speak_thread.start()
        
        if block:
            # Wait for speech to complete
            if self.speak_thread:
                self.speak_thread.join()
            # Double-check it's actually done
            with self.speak_lock:
                if self.is_speaking:
                    # Force mark as not speaking if thread finished but flag wasn't updated
                    self.is_speaking = False
    
    def wait_until_done(self):
        """Wait until current speech finishes."""
        if self.speak_thread and self.speak_thread.is_alive():
            print("⏳ Waiting for speech to finish...")
            self.speak_thread.join()
            print("✅ Speech finished")
    
    def is_currently_speaking(self) -> bool:
        """Check if currently speaking."""
        with self.speak_lock:
            return self.is_speaking
    
    def __del__(self):
        """Cleanup."""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass


class SpeechTranscriber:
    """Handles speech-to-text transcription."""
    
    def __init__(self, method="auto", language="en-US", transcription_timeout=5):
        """
        Initialize transcriber.
        
        Args:
            method: "whisper" (offline), "google" (online), or "auto" (try whisper first)
            language: Language code (default: "en-US")
            transcription_timeout: Timeout for transcription in seconds (default: 5)
        """
        self.method = method
        self.language = language
        self.transcription_timeout = transcription_timeout
        self.audio_buffer = deque(maxlen=200)  # Buffer for audio chunks
        self.target_chunk_count = 150  # Number of chunks to accumulate before transcribing
        self.use_whisper = False
        self.whisper_model = None
        self.recognizer = None
        self.last_transcription_time = 0
        self.failed_attempts = 0
        
        # Initialize Whisper if available and method is "whisper" or "auto"
        if method == "whisper" or (method == "auto" and WHISPER_AVAILABLE):
            if WHISPER_AVAILABLE:
                try:
                    print("🔄 Loading Whisper model (this may take a moment)...")
                    self.whisper_model = whisper.load_model("base")
                    self.use_whisper = True
                    print("✅ Whisper model loaded - offline transcription ready")
                except Exception as e:
                    print(f"⚠️  Failed to load Whisper model: {e}")
                    self.use_whisper = False
        
        # Fallback to Google if Whisper not available or method is "google"
        if method == "google" or (method == "auto" and not self.use_whisper):
            if SPEECH_RECOGNITION_AVAILABLE:
                self.recognizer = sr.Recognizer()
                # Adjust energy threshold for better recognition
                self.recognizer.energy_threshold = 300  # Lower = more sensitive
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8  # Pause detection
                self.use_whisper = False
                print("✅ Using Google Speech Recognition (requires internet)")
                print(f"   Energy threshold: {self.recognizer.energy_threshold}")
            else:
                print("⚠️  Speech recognition libraries not available")
                print("   Install with: pip install SpeechRecognition")
    
    def add_audio_chunk(self, audio_chunk: AudioChunk):
        """Add audio chunk to buffer for transcription."""
        self.audio_buffer.append(audio_chunk)
    
    def transcribe_buffer(self) -> Optional[str]:
        """
        Transcribe accumulated audio chunks.
        Accumulates exactly target_chunk_count chunks, then transcribes and clears.
        Returns transcribed text or None if no speech detected.
        Returns special marker string "__TRANSCRIPTION_ATTEMPTED__" if transcription was attempted but failed.
        """
        # Wait until we have exactly target_chunk_count chunks
        if len(self.audio_buffer) < self.target_chunk_count:
            return None
        
        print(f"📊 Buffer reached {len(self.audio_buffer)} chunks ({len(self.audio_buffer) * 1024 / 16000:.2f}s), transcribing...")
        
        # Get exactly target_chunk_count chunks from buffer
        chunks_to_transcribe = list(self.audio_buffer)[-self.target_chunk_count:]
        
        # Get all chunks (try both speaking and non-speaking, as long as there's volume)
        audio_data_list = []
        speaking_chunks = [chunk.data for chunk in chunks_to_transcribe if chunk.is_speaking]
        volume_chunks = [chunk.data for chunk in chunks_to_transcribe if chunk.volume > 0.01]
        
        # Prefer speaking chunks, but use volume chunks if no speaking chunks
        if len(speaking_chunks) > 0:
            audio_data_list = speaking_chunks
            print(f"   Using {len(speaking_chunks)} speaking chunks")
        elif len(volume_chunks) > 0:
            audio_data_list = volume_chunks
            print(f"   Using {len(volume_chunks)} chunks with volume > 0.01")
        else:
            # Use all chunks regardless of VAD
            audio_data_list = [chunk.data for chunk in chunks_to_transcribe]
            print(f"   Using all {len(chunks_to_transcribe)} chunks (VAD may have missed speech)")
        
        if len(audio_data_list) == 0:
            print("   No audio data to transcribe")
            return None
        
        # Concatenate audio chunks
        audio_data = np.concatenate(audio_data_list)
        sample_rate = self.audio_buffer[0].sample_rate
        
        # Calculate duration
        duration = len(audio_data) / sample_rate
        print(f"   Total audio: {len(audio_data)} samples = {duration:.2f} seconds")
        
        try:
            if self.use_whisper and self.whisper_model:
                # Whisper transcription (offline)
                print(f"🎤 Attempting Whisper transcription ({duration:.2f}s of audio)...")
                start_time = time.time()
                
                try:
                    result = self.whisper_model.transcribe(audio_data, language="en")
                    elapsed = time.time() - start_time
                    transcript = result["text"].strip()
                    
                    if transcript:
                        print(f"✅ Whisper transcription successful in {elapsed:.2f}s")
                        self.last_transcription_time = time.time()
                        self.failed_attempts = 0
                        # Clear buffer after successful transcription
                        self.audio_buffer.clear()
                        print("   ✅ Buffer cleared, ready for next 150 chunks")
                        return transcript
                    else:
                        print(f"⚠️  Whisper returned empty transcript ({elapsed:.2f}s)")
                        self.last_transcription_time = time.time()
                        # Clear buffer even on empty result
                        self.audio_buffer.clear()
                        print("   ✅ Buffer cleared after empty result, ready for next 150 chunks")
                        return "__TRANSCRIPTION_ATTEMPTED__"
                except Exception as e:
                    elapsed = time.time() - start_time
                    print(f"⚠️  Whisper transcription error after {elapsed:.2f}s: {e}")
                    return None
            elif self.recognizer:
                # Google Speech Recognition (requires internet)
                # Convert to AudioData format for speech_recognition
                audio_int16 = (audio_data * 32768.0).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                audio_source = sr.AudioData(
                    audio_bytes,
                    sample_rate,
                    2  # bytes per sample (16-bit = 2 bytes)
                )
                
                try:
                    print(f"🎤 Attempting Google transcription ({duration:.2f}s of audio)...")
                    start_time = time.time()
                    
                    # Use threading to implement timeout (since signal.alarm doesn't work well in threads)
                    result_container = {'transcript': None, 'error': None}
                    
                    def recognize():
                        try:
                            result_container['transcript'] = self.recognizer.recognize_google(
                                audio_source, 
                                language=self.language,
                                show_all=False
                            )
                        except Exception as e:
                            result_container['error'] = e
                    
                    # Run recognition in a thread with timeout
                    rec_thread = threading.Thread(target=recognize, daemon=True)
                    rec_thread.start()
                    rec_thread.join(timeout=self.transcription_timeout)
                    
                    elapsed = time.time() - start_time
                    
                    if rec_thread.is_alive():
                        print(f"⚠️  Google transcription timed out after {elapsed:.1f}s")
                        self.failed_attempts += 1
                        self.last_transcription_time = time.time()
                        # Clear buffer on timeout
                        self.audio_buffer.clear()
                        print("   ✅ Buffer cleared after timeout, ready for next 150 chunks")
                        if self.failed_attempts > 3:
                            print("⚠️  Too many failed attempts")
                            self.failed_attempts = 0
                        return "__TRANSCRIPTION_ATTEMPTED__"
                    
                    if result_container['error']:
                        error = result_container['error']
                        if isinstance(error, sr.UnknownValueError):
                            # Speech was unintelligible - this is normal
                            print(f"   (Speech not recognized - {elapsed:.1f}s)")
                            self.last_transcription_time = time.time()
                            # Clear buffer after failed attempt
                            self.audio_buffer.clear()
                            print("   ✅ Buffer cleared after 'speech not recognized', ready for next 150 chunks")
                            return "__TRANSCRIPTION_ATTEMPTED__"
                        elif isinstance(error, sr.RequestError):
                            print(f"⚠️  Speech recognition API error: {error}")
                            print("   (Check internet connection for Google Speech Recognition)")
                            self.failed_attempts += 1
                            self.last_transcription_time = time.time()
                            # Clear buffer on API error
                            self.audio_buffer.clear()
                            print("   ✅ Buffer cleared after API error, ready for next 150 chunks")
                            return "__TRANSCRIPTION_ATTEMPTED__"
                        else:
                            print(f"⚠️  Transcription error: {error}")
                            return None
                    
                    transcript = result_container['transcript']
                    self.last_transcription_time = time.time()
                    self.failed_attempts = 0  # Reset on success
                    
                    if transcript:
                        print(f"✅ Transcription successful in {elapsed:.2f}s")
                        self.last_transcription_time = time.time()
                        self.failed_attempts = 0
                        # Clear entire buffer after successful transcription
                        self.audio_buffer.clear()
                        print("   ✅ Buffer cleared, ready for next 150 chunks")
                        return transcript
                    else:
                        print(f"⚠️  Google returned empty transcript ({elapsed:.2f}s)")
                        self.last_transcription_time = time.time()
                        # Clear buffer even on empty result
                        self.audio_buffer.clear()
                        print("   ✅ Buffer cleared after empty result, ready for next 150 chunks")
                        return "__TRANSCRIPTION_ATTEMPTED__"
                        
                except Exception as e:
                    print(f"⚠️  Unexpected error in Google transcription: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
        except Exception as e:
            print(f"⚠️  Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return None


class AudioRecorder:
    """Records audio in real-time."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, device_index=None):
        """
        Initialize audio recorder.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_size: Number of frames per buffer
            device_index: Audio device index (None for default)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Voice activity detection parameters
        self.vad_threshold = 0.01  # RMS threshold for detecting speech
        self.volume_history = deque(maxlen=30)  # History for smoothing
    
    def calculate_rms(self, audio_data: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) volume of audio chunk."""
        return np.sqrt(np.mean(audio_data**2))
    
    def detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Simple voice activity detection based on RMS volume."""
        rms = self.calculate_rms(audio_data)
        self.volume_history.append(rms)
        
        # Use smoothed average
        if len(self.volume_history) > 5:
            avg_volume = np.mean(list(self.volume_history))
            return rms > self.vad_threshold and rms > avg_volume * 0.5
        return rms > self.vad_threshold
    
    def start_stream(self):
        """Start audio stream."""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_index
        )
        self.stream.start_stream()
        print(f"✅ Audio recording started: {self.sample_rate}Hz, chunk_size={self.chunk_size}")
    
    def read_chunk(self) -> Optional[AudioChunk]:
        """
        Read a chunk of audio data (BLOCKING method).
        
        Returns:
            AudioChunk with audio data, timestamp, and VAD results, or None if error
        """
        if not self.stream or not self.stream.is_active():
            return None
        
        try:
            # Read audio data (blocking call)
            audio_bytes = self.stream.read(self.chunk_size, exception_on_overflow=False)
            
            # Convert to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1, 1] range
            audio_data = audio_data / 32768.0
            
            # Calculate RMS volume
            volume = self.calculate_rms(audio_data)
            
            # Detect voice activity
            is_speaking = self.detect_voice_activity(audio_data)
            
            # Create AudioChunk
            chunk = AudioChunk(
                data=audio_data,
                timestamp=time.time(),
                sample_rate=self.sample_rate,
                is_speaking=is_speaking,
                volume=volume
            )
            
            return chunk
        except Exception as e:
            print(f"⚠️  Error reading audio chunk: {e}")
            return None
    
    def stop_stream(self):
        """Stop audio stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
    
    def __del__(self):
        """Cleanup."""
        self.stop_stream()


def main():
    """Main function - audio-only analysis."""
    print("=" * 60)
    print("Audio-Only Analysis with Transcription and AI Responses")
    print("=" * 60)
    
    # Thread control
    running = threading.Event()
    running.set()  # Start as running
    
    # Initialize audio recorder
    recorder = AudioRecorder(sample_rate=16000, chunk_size=1024)
    recorder.start_stream()
    
    # Initialize text-to-speech
    TTS_METHOD = "auto"
    ENABLE_TTS = True
    tts = None
    if ENABLE_TTS:
        print("🔄 Initializing text-to-speech...")
        tts = TextToSpeech(method=TTS_METHOD)
        if tts.use_pyttsx3 or GTTS_AVAILABLE:
            print("✅ Text-to-speech enabled")
    
    # Initialize transcriber
    TRANSCRIPTION_METHOD = "whisper"  # Options: "whisper", "google", or "auto"
    transcriber = None
    if TRANSCRIPTION_AVAILABLE:
        print(f"🔄 Initializing transcription with method: {TRANSCRIPTION_METHOD}...")
        transcriber = SpeechTranscriber(method=TRANSCRIPTION_METHOD, language="en-US")
        if transcriber.use_whisper:
            print("✅ Transcription enabled - Using OpenAI Whisper (offline)")
        elif transcriber.recognizer:
            print("✅ Transcription enabled - Using Google Speech Recognition (online)")
        else:
            print("⚠️  Transcription initialized but no recognizer available")
            transcriber = None
    else:
        print("⚠️  Transcription libraries not available")
        print("   Install: pip install SpeechRecognition")
        print("   Or: pip install openai-whisper")
    
    print("\n" + "=" * 60)
    print("Ready! Start speaking... (Press Ctrl+C to stop)")
    print("=" * 60 + "\n")
    
    chunk_count = 0
    ai_processing = False  # Track if AI is currently processing a response
    last_ai_response_time = 0  # Track when last AI response was sent
    
    try:
        while running.is_set():
            chunk = recorder.read_chunk()
            if chunk is None:
                continue
            
            chunk_count += 1
            
            # CRITICAL: Don't process ANY audio while TTS is speaking or AI is processing
            if tts and tts.is_currently_speaking():
                # Completely ignore audio while TTS is speaking
                # Don't even add to buffer - we don't care what user says while AI is talking
                continue
            
            # Also ignore audio while AI is processing (waiting for response)
            if ai_processing:
                # Don't add to buffer - wait for current AI response to complete
                continue
            
            # Add to transcriber buffer (only if not speaking and not processing)
            if transcriber:
                transcriber.add_audio_chunk(chunk)
                # Try to transcribe accumulated audio
                transcript = transcriber.transcribe_buffer()
                
                # Buffer is already cleared inside transcribe_buffer() after transcription attempt
                # Just convert marker back to None if needed
                if transcript == "__TRANSCRIPTION_ATTEMPTED__":
                    transcript = None  # Convert marker back to None
                
                if transcript:
                    chunk.transcript = transcript
                    
                    # Print to console
                    print(f"📝 Transcript: {transcript}")
                    
                    # AI API configuration
                    AI_API_URL = "http://localhost:11434/api/generate"  # Ollama endpoint
                    AI_MODEL = "llama3.2:latest"  # Change this to your model name (check with: ollama list)
                    ENABLE_AI_RESPONSE = True  # Set to False to disable AI responses
                    
                    full_response = ""
                    if ENABLE_AI_RESPONSE:
                        # CRITICAL: Don't start new AI call if one is already processing
                        if ai_processing:
                            print("⏳ AI is already processing a response, skipping...")
                            continue
                        
                        # CRITICAL: Wait for TTS to finish before starting new AI call
                        if tts and tts.is_currently_speaking():
                            print("⏳ TTS is still speaking, waiting for it to finish...")
                            tts.wait_until_done()
                            print("✅ TTS finished, now calling AI...")
                        
                        # Mark as processing - prevents multiple AI calls
                        ai_processing = True
                        last_ai_response_time = time.time()
                        
                        # Run AI API call in separate try-except to prevent crashes
                        try:
                            print(f"🤖 Calling AI API: {AI_MODEL}...")
                            
                            # Make request with timeout and error handling
                            response = requests.post(
                                AI_API_URL,
                                json={
                                    "model": AI_MODEL,
                                    "prompt": INITIAL_PROMPT + transcript,
                                    "stream": True
                                },
                                stream=True,
                                timeout=30,  # 30 second timeout
                                headers={'Content-Type': 'application/json'}
                            )
                            
                            # Check for errors
                            response.raise_for_status()
                            
                            # Collect the FULL response - wait until "done": true using while loop
                            print("   Collecting complete response...")
                            response_complete = False
                            response_buffer = []  # Buffer for display after collection
                            
                            # WHILE LOOP: Keep reading until response is complete (done=True)
                            while not response_complete:
                                # Read lines from stream
                                for line in response.iter_lines():
                                    if not line:
                                        continue
                                    
                                    try:
                                        data = json.loads(line)
                                        
                                        # Accumulate response text (silently)
                                        if "response" in data:
                                            response_text = data["response"]
                                            full_response += response_text
                                            response_buffer.append(response_text)
                                        
                                        # CRITICAL: Check if response is complete - wait for this!
                                        if data.get("done", False):
                                            response_complete = True  # Exit while loop
                                            # Now print the complete response
                                            print("\n✅ AI response complete:")
                                            print("".join(response_buffer))
                                            break  # Exit for loop, which exits while loop
                                        
                                        # Check for errors
                                        elif "error" in data:
                                            print(f"\n⚠️  AI API error: {data['error']}")
                                            response_complete = True  # Exit while loop
                                            break  # Exit for loop
                                            
                                    except json.JSONDecodeError:
                                        continue
                                
                                # If we exit the for loop without done=True, the stream ended
                                # Check if we have response text - if yes, treat as complete
                                if not response_complete:
                                    if full_response.strip():
                                        # Stream ended but we have response - treat as complete
                                        print("\n⚠️  Stream ended without 'done' flag, treating as complete")
                                        response_complete = True
                                        print("✅ AI response complete:")
                                        print("".join(response_buffer))
                                    else:
                                        # No response at all
                                        print("\n⚠️  Stream ended with no response")
                                        response_complete = True  # Exit while loop
                                    break  # Exit while loop
                            
                            # Only speak if response is complete
                            if response_complete and full_response.strip():
                                if tts:
                                    print("\n🔊 Speaking complete response (will not be interrupted)...")
                                    # CRITICAL: Use block=True to WAIT for TTS to finish completely
                                    # This ensures TTS finishes before anything else happens
                                    ai_processing = False  # Mark AI as done before TTS
                                    tts.speak(full_response.strip(), block=True, wait_if_busy=True)
                                    
                                    # WHILE LOOP: Double-check TTS is completely finished
                                    # Keep waiting until TTS is definitely done
                                    print("   Waiting for TTS to completely finish...")
                                    while tts.is_currently_speaking():
                                        time.sleep(0.1)  # Check every 100ms
                                    
                                    # ADDITIONAL DELAY: Wait a bit longer to ensure audio actually finishes
                                    # Sometimes is_speaking becomes False before audio completes
                                    print("   Adding safety delay to ensure audio playback completes...")
                                    time.sleep(0.5)  # Wait 500ms after TTS says it's done
                                    
                                    # Final check - make sure it's still not speaking
                                    if tts.is_currently_speaking():
                                        print("   ⚠️  TTS still speaking after delay, waiting more...")
                                        while tts.is_currently_speaking():
                                            time.sleep(0.1)
                                        time.sleep(0.3)  # Additional delay after it actually stops
                                    
                                    print("✅ TTS completely finished, ready for next transcript")
                                else:
                                    print("\n✅ AI response complete (TTS disabled)")
                                    ai_processing = False  # Mark as done
                            elif not response_complete:
                                print("\n⚠️  AI response incomplete or interrupted - not speaking")
                                ai_processing = False  # Mark as done even on error
                            elif not full_response.strip():
                                print("\n⚠️  No response text from AI - not speaking")
                                ai_processing = False  # Mark as done
                            else:
                                print("\n⚠️  Unexpected state - not speaking")
                                ai_processing = False  # Mark as done
                        
                        except requests.exceptions.ConnectionError as e:
                            print(f"\n❌ Cannot connect to AI API at {AI_API_URL}")
                            print(f"   Error: {e}")
                            print("   Make sure Ollama is running: ollama serve")
                            print("   Or check if the URL is correct")
                            ai_processing = False  # Mark as done on error
                        
                        except requests.exceptions.HTTPError as e:
                            print(f"\n❌ AI API HTTP error: {e}")
                            if e.response:
                                if e.response.status_code == 404:
                                    print("   Endpoint not found. Try:")
                                    print("   - Check if Ollama is running")
                                    print("   - Verify the API URL is correct")
                                    print("   - Check if the model name is correct")
                                elif e.response.status_code == 400:
                                    print("   Bad request. Check your prompt format.")
                            else:
                                print("   Unknown HTTP error")
                            ai_processing = False  # Mark as done on error
                        
                        except requests.exceptions.Timeout:
                            print(f"\n❌ AI API timeout (>30s)")
                            ai_processing = False  # Mark as done on timeout
                        
                        except KeyboardInterrupt:
                            print("\n⚠️  AI API call interrupted by user")
                            ai_processing = False  # Mark as done on interrupt
                            # Don't crash, just continue
                        
                        except Exception as e:
                            # Catch ALL exceptions to prevent crashes
                            print(f"\n❌ AI API error: {e}")
                            print("   (Continuing without AI response to prevent crash)")
                            ai_processing = False  # Mark as done on error
            
            # Print audio stats for debugging (every 100 chunks)
            if chunk_count % 100 == 0:
                if chunk.is_speaking:
                    print(f"   📊 Audio stats: Volume={chunk.volume:.3f}, Speaking={chunk.is_speaking}, Chunks={chunk_count}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        running.clear()
    
    finally:
        # Cleanup
        recorder.stop_stream()
        print("\n✅ Audio recorder stopped")
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()

