"""
Multi-threaded Video and Audio Analysis with Speech-to-Text Transcription
Runs video processing and audio recording in separate threads that communicate.

FEATURES:
- ✅ Video thread: Processes camera frames with face detection/tracking
- ✅ Audio thread: Records audio and detects voice activity
- ✅ Transcription: Real-time speech-to-text conversion (English)
- ✅ Communication: Both threads share data via queues and shared state
- ✅ Synchronization: Timestamps help correlate audio and video events
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
import threading
from queue import Queue, Empty
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
INITIAL_PROMPT = """
You are a helpful assistant that can answer questions and help with tasks.
You are currently in a cocktail party.
You are sitting at a table with a group of people.
You are listening to the conversation and trying to understand what is going on.
You are also trying to say something interesting to the group or show interest in the conversation.
Here's what the group is talking about:
"""
# Try to import face_recognition, fallback to basic if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition library available - using deep facial features")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  face_recognition not installed - using MediaPipe features")
    print("   For better tracking: pip install face-recognition")

# Try to import mss for screen capture, fallback if not available
try:
    import mss
    SCREEN_CAPTURE_AVAILABLE = True
except ImportError:
    SCREEN_CAPTURE_AVAILABLE = False
    print("⚠️  mss not installed - screen capture not available")
    print("   For screen sharing: pip install mss")

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
class Face:
    """Represents a detected face in the video."""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]
    is_talking: bool = False
    confidence: float = 0.0
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    name: Optional[str] = None
    timestamp: float = 0.0  # When this face was detected


@dataclass
class AudioChunk:
    """Represents an audio chunk from the microphone."""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    is_speaking: bool = False  # Voice activity detection result
    volume: float = 0.0  # RMS volume
    transcript: Optional[str] = None  # Transcribed text (if available)


@dataclass
class SharedState:
    """Shared state between video and audio threads."""
    def __init__(self):
        # Thread control
        self.running = threading.Event()
        self.running.set()  # Start as running
        
        # Video data
        self.video_queue = Queue(maxsize=10)  # Latest frames
        self.faces_data = {}  # {face_id: latest_face_info}
        self.faces_lock = threading.Lock()
        self.video_timestamp = 0.0
        
        # Audio data
        self.audio_queue = Queue(maxsize=10)  # Latest audio chunks
        self.audio_active = False  # Voice activity
        self.audio_volume = 0.0
        self.audio_timestamp = 0.0
        self.audio_lock = threading.Lock()
        
        # Transcription data
        self.transcript_queue = Queue(maxsize=20)  # Recent transcripts
        self.current_transcript = ""  # Latest transcript text
        self.transcript_history = deque(maxlen=10)  # Last 10 transcripts
        self.transcript_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'video_frames': 0,
            'audio_chunks': 0,
            'sync_events': 0
        }
        self.stats_lock = threading.Lock()
        
        # Communication events
        self.video_ready = threading.Event()
        self.audio_ready = threading.Event()


class FaceDetector:
    """Handles face detection and embedding extraction."""
    
    def __init__(self, min_detection_confidence=0.7):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=1
        )
        self.use_face_recognition = FACE_RECOGNITION_AVAILABLE
    
    def extract_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract facial embedding from face region."""
        x, y, w, h = bbox
        
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return None
        
        if self.use_face_recognition:
            rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_roi)
            if encodings:
                return encodings[0]
        else:
            resized = cv2.resize(face_roi, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            normalized = gray.flatten() / 255.0
            embedding = np.random.randn(128) * 0.1 + normalized[:128]
            return embedding
        
        return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Face]:
        """Detect faces and extract embeddings."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for idx, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                center_x = x + width // 2
                center_y = y + height // 2
                
                embedding = self.extract_embedding(frame, (x, y, width, height))
                
                face = Face(
                    id=idx,
                    bbox=(x, y, width, height),
                    center=(center_x, center_y),
                    confidence=detection.score[0],
                    embedding=embedding,
                    timestamp=time.time()
                )
                faces.append(face)
        
        return faces
    
    def __del__(self):
        self.face_detection.close()


class FaceTracker:
    """Simple face tracker that assigns persistent IDs."""
    
    def __init__(self, max_distance=150):
        self.tracked_faces: Dict[int, Dict] = {}
        self.next_id = 0
        self.max_distance = max_distance
    
    def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def update(self, new_detections: List[Face]) -> List[Face]:
        """Update tracking with new detections."""
        # Mark all as missing initially
        for tracked_id in self.tracked_faces:
            self.tracked_faces[tracked_id]['matched'] = False
        
        # Try to match new detections with existing tracks
        for detection in new_detections:
            best_match_id = None
            min_distance = float('inf')
            
            for tracked_id, tracked_data in self.tracked_faces.items():
                if tracked_data['matched']:
                    continue
                
                distance = self.calculate_distance(detection.center, tracked_data['center'])
                if distance < self.max_distance and distance < min_distance:
                    min_distance = distance
                    best_match_id = tracked_id
            
            if best_match_id is not None:
                # Update existing track
                detection.id = best_match_id
                self.tracked_faces[best_match_id]['center'] = detection.center
                self.tracked_faces[best_match_id]['bbox'] = detection.bbox
                self.tracked_faces[best_match_id]['matched'] = True
            else:
                # New face - assign new ID
                detection.id = self.next_id
                self.tracked_faces[self.next_id] = {
                    'center': detection.center,
                    'bbox': detection.bbox,
                    'matched': True
                }
                self.next_id += 1
        
        # Remove unmatched tracks
        ids_to_remove = [
            tracked_id for tracked_id, data in self.tracked_faces.items()
            if not data['matched']
        ]
        for tracked_id in ids_to_remove:
            del self.tracked_faces[tracked_id]
        
        return new_detections


class MouthMovementDetector:
    """Detects mouth movement using MediaPipe Face Mesh."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mouth_history: Dict[int, deque] = {}
        self.history_length = 10
    
    def calculate_mouth_aspect_ratio(self, landmarks, frame_shape) -> float:
        """Calculate mouth aspect ratio (MAR) to detect talking."""
        h, w = frame_shape[:2]
        
        upper = landmarks[13]
        lower = landmarks[14]
        left = landmarks[61]
        right = landmarks[291]
        
        upper_y = upper.y * h
        lower_y = lower.y * h
        left_x = left.x * w
        right_x = right.x * w
        
        vertical = abs(lower_y - upper_y)
        horizontal = abs(right_x - left_x)
        
        mar = vertical / (horizontal + 1e-6)
        return mar
    
    def detect_talking(self, frame: np.ndarray, face: Face, threshold=0.05) -> bool:
        """Detect if a person is talking based on mouth movement."""
        x, y, w, h = face.bbox
        
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return False
        
        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_roi)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            mar = self.calculate_mouth_aspect_ratio(landmarks, face_roi.shape)
            
            if face.id not in self.mouth_history:
                self.mouth_history[face.id] = deque(maxlen=self.history_length)
            
            self.mouth_history[face.id].append(mar)
            
            if len(self.mouth_history[face.id]) >= 5:
                variance = np.var(list(self.mouth_history[face.id]))
                return variance > threshold
        
        return False
    
    def __del__(self):
        self.face_mesh.close()


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
            self.speak_thread.join()
    
    def stop(self):
        """Stop any ongoing speech (use with caution - may interrupt)."""
        print("⚠️  Stopping TTS (may interrupt current speech)...")
        if self.use_pyttsx3 and self.engine:
            try:
                self.engine.stop()
            except:
                pass
        elif GTTS_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except:
                pass
        
        with self.speak_lock:
            self.is_speaking = False
            # Clear queue
            while not self.speak_queue.empty():
                try:
                    self.speak_queue.get_nowait()
                except:
                    pass
    
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
    """Handles speech-to-text transcription from audio chunks."""
    
    def __init__(self, method="auto", language="en-US", transcription_timeout=5.0):
        """
        Initialize speech transcriber.
        
        Args:
            method: "whisper" (offline), "google" (online), or "auto" (try whisper first)
            language: Language code (e.g., "en-US" for English)
            transcription_timeout: Timeout in seconds for transcription API calls (default: 5.0)
        """
        self.method = method
        self.language = language
        self.recognizer = None
        self.whisper_model = None
        self.use_whisper = False
        self.transcription_timeout = transcription_timeout
        
        # Audio buffer for transcription - use chunk count instead of time
        self.audio_buffer = deque(maxlen=200)  # Keep buffer larger than target
        self.target_chunk_count = 50  # Accumulate 50 chunks before transcribing
        self.last_transcription_time = time.time()
        self.failed_attempts = 0  # Track failed attempts
        
        # Initialize recognition method
        if method == "whisper" or (method == "auto" and WHISPER_AVAILABLE):
            if WHISPER_AVAILABLE:
                try:
                    print("🔄 Loading Whisper model (this may take a moment)...")
                    # Whisper model sizes: tiny, base, small, medium, large
                    # - tiny: fastest, least accurate
                    # - base: good balance (default)
                    # - small: better accuracy, slower
                    # - medium/large: best accuracy, slowest
                    WHISPER_MODEL_SIZE = "base"  # Change to "tiny", "small", "medium", or "large"
                    self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
                    self.use_whisper = True
                    print(f"✅ Whisper model '{WHISPER_MODEL_SIZE}' loaded")
                except Exception as e:
                    print(f"⚠️  Failed to load Whisper: {e}")
                    self.use_whisper = False
            else:
                print("⚠️  Whisper not available, install with: pip install openai-whisper")
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
                # Whisper expects audio as float32 array (values between -1 and 1)
                # audio_data is already in float32 format (normalized to [-1, 1])
                print(f"🎤 Attempting Whisper transcription ({len(audio_data)/sample_rate:.2f}s of audio)...")
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
                    print(f"🎤 Attempting Google transcription ({len(audio_data)/sample_rate:.2f}s of audio)...")
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
    
    def transcribe_chunk(self, audio_chunk: AudioChunk) -> Optional[str]:
        """
        Transcribe a single audio chunk (quick method for short utterances).
        For better results, use transcribe_buffer() with accumulated chunks.
        """
        if not audio_chunk.is_speaking:
            return None
        
        # For single chunks, only try if volume is high enough
        if audio_chunk.volume < 0.02:
            return None
        
        try:
            if self.use_whisper and self.whisper_model:
                # Whisper needs more data, so skip single chunks
                return None
            elif self.recognizer:
                # Convert to AudioData format
                audio_int16 = (audio_chunk.data * 32768.0).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                audio_source = sr.AudioData(
                    audio_bytes,
                    audio_chunk.sample_rate,
                    2
                )
                
                try:
                    transcript = self.recognizer.recognize_google(audio_source, language=self.language)
                    return transcript
                except (sr.UnknownValueError, sr.RequestError):
                    return None
        except Exception as e:
            return None
        
        return None


class AudioRecorder:
    """Records audio in a separate thread."""
    
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
        
        This method blocks until the chunk_size buffer is filled.
        In a loop, this automatically gives you real-time chunks as they become available.
        
        Returns:
            AudioChunk with audio data, timestamp, and VAD results, or None if error
        """
        if not self.stream or not self.stream.is_active():
            return None
        
        try:
            # BLOCKING READ: Waits until chunk_size samples are available
            # This automatically handles real-time streaming - when you call this
            # in a loop, it waits for each chunk to be ready
            audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            
            # Convert bytes to numpy array
            # audio_data is raw bytes, convert to int16 array, then to float32 [-1.0, 1.0]
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Process audio
            is_speaking = self.detect_voice_activity(audio_array)
            volume = self.calculate_rms(audio_array)
            
            return AudioChunk(
                data=audio_array,
                timestamp=time.time(),
                sample_rate=self.sample_rate,
                is_speaking=is_speaking,
                volume=volume
            )
        except Exception as e:
            print(f"Audio read error: {e}")
            return None
    
    def read_chunk_non_blocking(self) -> Optional[AudioChunk]:
        """
        Read a chunk of audio data (NON-BLOCKING method).
        
        Returns immediately with available data, or None if no data ready.
        Use this if you don't want to block waiting for chunks.
        
        Returns:
            AudioChunk or None if no data available
        """
        if not self.stream or not self.stream.is_active():
            return None
        
        try:
            # Get available frames (might be less than chunk_size)
            frames_available = self.stream.get_read_available()
            if frames_available < self.chunk_size:
                return None  # Not enough data yet
            
            # Read only what's available
            audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            is_speaking = self.detect_voice_activity(audio_array)
            volume = self.calculate_rms(audio_array)
            
            return AudioChunk(
                data=audio_array,
                timestamp=time.time(),
                sample_rate=self.sample_rate,
                is_speaking=is_speaking,
                volume=volume
            )
        except Exception as e:
            print(f"Audio read error: {e}")
            return None
    
    def stop_stream(self):
        """Stop audio stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("✅ Audio recording stopped")


class ScreenCapture:
    """Captures screen frames for video analysis."""
    
    def __init__(self, monitor_number=1, target_fps=30):
        if not SCREEN_CAPTURE_AVAILABLE:
            raise ImportError("mss library not installed. Install with: pip install mss")
        
        self.mss = mss.mss()
        self.monitor_number = monitor_number
        
        if monitor_number == 0:
            self.monitor = self.mss.monitors[0]
        else:
            if monitor_number >= len(self.mss.monitors):
                raise ValueError(f"Monitor {monitor_number} not available.")
            self.monitor = self.mss.monitors[monitor_number]
        
        self.width = self.monitor['width']
        self.height = self.monitor['height']
    
    def read(self):
        try:
            screenshot = self.mss.grab(self.monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return True, frame
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return False, None
    
    def isOpened(self):
        return SCREEN_CAPTURE_AVAILABLE
    
    def release(self):
        if hasattr(self, 'mss'):
            self.mss.close()


def video_thread(video_source, shared_state: SharedState):
    """Video processing thread."""
    print("📹 Video thread starting...")
    
    # Initialize video capture
    if isinstance(video_source, str) and video_source.lower().startswith("screen"):
        if not SCREEN_CAPTURE_AVAILABLE:
            print("❌ Screen capture not available")
            return
        cap = ScreenCapture(monitor_number=1, target_fps=30)
    else:
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Could not open video source: {video_source}")
        return
    
    # Initialize detectors (each detector manages its own MediaPipe objects)
    # MediaPipe objects are thread-safe if used within the same thread
    try:
        face_detector = FaceDetector()
        face_tracker = FaceTracker()
        mouth_detector = MouthMovementDetector()
    except Exception as e:
        print(f"❌ Failed to initialize detectors: {e}")
        cap.release()
        return
    
    frame_count = 0
    shared_state.video_ready.set()
    
    try:
        while shared_state.running.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Detect faces
            detected_faces = face_detector.detect_faces(frame)
            
            # Track faces
            tracked_faces = face_tracker.update(detected_faces)
            
            # Detect talking
            for face in tracked_faces:
                face.is_talking = mouth_detector.detect_talking(frame, face)
                face.timestamp = current_time
            
            # Update shared state
            with shared_state.faces_lock:
                shared_state.faces_data = {
                    face.id: {
                        'bbox': face.bbox,
                        'center': face.center,
                        'is_talking': face.is_talking,
                        'confidence': face.confidence,
                        'name': face.name,
                        'timestamp': face.timestamp
                    }
                    for face in tracked_faces
                }
                shared_state.video_timestamp = current_time
            
            # Put frame in queue (non-blocking, drop old frames)
            try:
                shared_state.video_queue.put_nowait((frame.copy(), tracked_faces.copy(), current_time))
            except:
                # Queue full, remove oldest and add new
                try:
                    shared_state.video_queue.get_nowait()
                    shared_state.video_queue.put_nowait((frame.copy(), tracked_faces.copy(), current_time))
                except:
                    pass
            
            # Update stats
            with shared_state.stats_lock:
                shared_state.stats['video_frames'] = frame_count
            
            # Small delay to prevent overwhelming
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("⚠️  Video thread interrupted")
    except Exception as e:
        print(f"❌ Video thread error: {e}")
        # Don't print full traceback for MediaPipe/OpenCV TLS errors - they're common
        if "TLS" not in str(e) and "OpenCV" not in str(e):
            import traceback
            traceback.print_exc()
    
    finally:
        # Cleanup MediaPipe objects properly
        try:
            if 'face_detector' in locals():
                face_detector.face_detection.close()
        except:
            pass
        try:
            if 'mouth_detector' in locals():
                mouth_detector.face_mesh.close()
        except:
            pass
        
        cap.release()
        cv2.destroyAllWindows()
        shared_state.video_ready.clear()
        print("📹 Video thread stopped")


def audio_thread(shared_state: SharedState, save_file=None, enable_transcription=True):
    """Audio recording thread with transcription."""
    print("🎤 Audio thread starting...")
    
    recorder = AudioRecorder(sample_rate=16000, chunk_size=1024)
    recorder.start_stream()
    
    # Initialize text-to-speech for reading transcripts
    # TTS_METHOD options: "pyttsx3" (offline), "gtts" (online), or "auto"
    TTS_METHOD = "auto"
    ENABLE_TTS = True  # Set to False to disable text-to-speech
    
    tts = None
    if ENABLE_TTS:
        print("🔄 Initializing text-to-speech...")
        tts = TextToSpeech(method=TTS_METHOD)
        if tts.use_pyttsx3 or GTTS_AVAILABLE:
            print("✅ Text-to-speech enabled")
    
    # Initialize transcriber if available
    # TRANSCRIPTION_METHOD options:
    #   "whisper" - Use OpenAI Whisper (offline, better accuracy, slower)
    #   "google" - Use Google Speech Recognition (online, requires internet)
    #   "auto" - Try Whisper first, fallback to Google
    TRANSCRIPTION_METHOD = "whisper"  # Change this to "google", "whisper", or "auto"
    
    transcriber = None
    if enable_transcription and TRANSCRIPTION_AVAILABLE:
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
        if not enable_transcription:
            print("⚠️  Transcription disabled")
        elif not TRANSCRIPTION_AVAILABLE:
            print("⚠️  Transcription libraries not available")
            print("   Install: pip install SpeechRecognition")
            print("   Or: pip install openai-whisper")
    
    audio_file = None
    if save_file:
        audio_file = wave.open(save_file, 'wb')
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)  # 16-bit
        audio_file.setframerate(16000)
    
    chunk_count = 0
    shared_state.audio_ready.set()
    
    try:
        while shared_state.running.is_set():
            chunk = recorder.read_chunk()
            if chunk is None:
                continue
            
            chunk_count += 1
            
            # Add to transcriber buffer
            if transcriber:
                # Don't transcribe while TTS is speaking - let AI finish talking
                if tts and tts.is_currently_speaking():
                    # Skip transcription while speaking - just add to buffer
                    transcriber.add_audio_chunk(chunk)
                else:
                    transcriber.add_audio_chunk(chunk)
                    # Try to transcribe accumulated audio
                    transcript = transcriber.transcribe_buffer()
                
                # Buffer is already cleared inside transcribe_buffer() after transcription attempt
                # Just convert marker back to None if needed
                if transcript == "__TRANSCRIPTION_ATTEMPTED__":
                    transcript = None  # Convert marker back to None
                
                if transcript:
                    chunk.transcript = transcript
                    
                    # Update shared state with transcript
                    with shared_state.transcript_lock:
                        shared_state.current_transcript = transcript
                        shared_state.transcript_history.append({
                            'text': transcript,
                            'timestamp': chunk.timestamp
                        })
                        
                        # Put in transcript queue
                        try:
                            shared_state.transcript_queue.put_nowait({
                                'text': transcript,
                                'timestamp': chunk.timestamp
                            })
                        except:
                            # Queue full, remove oldest
                            try:
                                shared_state.transcript_queue.get_nowait()
                                shared_state.transcript_queue.put_nowait({
                                    'text': transcript,
                                    'timestamp': chunk.timestamp
                                })
                            except:
                                pass
                    
                    # Print to console
                    print(f"📝 Transcript: {transcript}")

                    # AI API configuration
                    AI_API_URL = "http://localhost:11434/api/generate"  # Ollama endpoint
                    AI_MODEL = "llama3.2:latest"  # Change this to your model name (check with: ollama list)
                    ENABLE_AI_RESPONSE = True  # Set to False to disable AI responses
                    
                    full_response = ""
                    if ENABLE_AI_RESPONSE:
                        # CRITICAL: Wait for TTS to finish before starting new AI call
                        if tts and tts.is_currently_speaking():
                            print("⏳ TTS is still speaking, waiting for it to finish...")
                            tts.wait_until_done()
                            print("✅ TTS finished, now calling AI...")
                        
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
                            response.raise_for_status()  # Raises exception for bad status codes
                            
                            # Collect the FULL response - wait until "done": true using while loop
                            # Don't print until complete to avoid interleaving with TTS messages
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
                                            response_buffer.append(response_text)  # Store for later display
                                        
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
                                    # Use block=False to not block audio thread, but wait_if_busy=True
                                    # This ensures TTS queues properly and finishes before next AI call
                                    tts.speak(full_response.strip(), block=False, wait_if_busy=True)
                                    # Wait for TTS to finish (non-blocking check in loop)
                                    # This ensures no new transcriptions start until TTS completes
                                    print("   (TTS started, will complete before next response)")
                                else:
                                    print("\n✅ AI response complete (TTS disabled)")
                            elif not response_complete:
                                print("\n⚠️  AI response incomplete or interrupted - not speaking")
                            elif not full_response.strip():
                                print("\n⚠️  No response text from AI - not speaking")
                            else:
                                print("\n⚠️  Unexpected state - not speaking")
                        
                        except requests.exceptions.ConnectionError as e:
                            print(f"\n❌ Cannot connect to AI API at {AI_API_URL}")
                            print(f"   Error: {e}")
                            print("   Make sure Ollama is running: ollama serve")
                            print("   Or check if the URL is correct")
                        
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
                        
                        except requests.exceptions.Timeout:
                            print(f"\n❌ AI API timeout (>30s)")
                        
                        except KeyboardInterrupt:
                            print("\n⚠️  AI API call interrupted by user")
                            # Don't crash, just continue
                        
                        except Exception as e:
                            # Catch ALL exceptions to prevent crashes
                            print(f"\n❌ AI API error: {e}")
                            print("   (Continuing without AI response to prevent crash)")
                            # Don't print full traceback to avoid cluttering output
                            # import traceback
                            # traceback.print_exc()
                    else:
                        print("   (AI responses disabled)")

                    # Also print audio stats for debugging
                    if chunk.is_speaking:
                        print(f"   (Volume: {chunk.volume:.3f}, Speaking: {chunk.is_speaking})")
            
            # Debug output showing progress toward 150 chunks
            if transcriber and chunk_count % 50 == 0:
                buffer_size = len(transcriber.audio_buffer)
                speaking_chunks = sum(1 for c in transcriber.audio_buffer if c.is_speaking)
                progress = (buffer_size / transcriber.target_chunk_count) * 100
                print(f"📊 Buffer progress: {buffer_size}/{transcriber.target_chunk_count} chunks ({progress:.1f}%), "
                      f"Speaking={speaking_chunks}, Volume={chunk.volume:.3f}")
            
            # Update shared state
            with shared_state.audio_lock:
                shared_state.audio_active = chunk.is_speaking
                shared_state.audio_volume = chunk.volume
                shared_state.audio_timestamp = chunk.timestamp
            
            # Put chunk in queue (non-blocking)
            try:
                shared_state.audio_queue.put_nowait(chunk)
            except:
                try:
                    shared_state.audio_queue.get_nowait()
                    shared_state.audio_queue.put_nowait(chunk)
                except:
                    pass
            
            # Save to file if requested
            if audio_file:
                audio_int16 = (chunk.data * 32768.0).astype(np.int16)
                audio_file.writeframes(audio_int16.tobytes())
            
            # Update stats
            with shared_state.stats_lock:
                shared_state.stats['audio_chunks'] = chunk_count
            
            # Small delay
            time.sleep(0.001)
    
    except Exception as e:
        print(f"❌ Audio thread error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        recorder.stop_stream()
        if audio_file:
            audio_file.close()
        shared_state.audio_ready.clear()
        print("🎤 Audio thread stopped")


def display_thread(shared_state: SharedState):
    """Display thread that shows synchronized video and audio data."""
    print("🖥️  Display thread starting...")
    
    cv2.namedWindow('Multi-threaded Video + Audio', cv2.WINDOW_NORMAL)
    
    try:
        while shared_state.running.is_set():
            # Get latest video frame
            try:
                frame, faces, frame_time = shared_state.video_queue.get(timeout=0.1)
            except Empty:
                continue
            
            # Get latest audio state
            with shared_state.audio_lock:
                audio_active = shared_state.audio_active
                audio_volume = shared_state.audio_volume
                audio_time = shared_state.audio_timestamp
            
            # Get latest faces data
            with shared_state.faces_lock:
                faces_data = shared_state.faces_data.copy()
                latest_video_time = shared_state.video_timestamp
            
            # Get latest transcript
            with shared_state.transcript_lock:
                current_transcript = shared_state.current_transcript
                transcript_history = list(shared_state.transcript_history)
            
            # Annotate frame
            annotated = frame.copy()
            
            # Draw faces
            for face_id, face_info in faces_data.items():
                x, y, w, h = face_info['bbox']
                is_talking = face_info['is_talking']
                
                color = (0, 255, 0) if is_talking else (255, 0, 0)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                
                label = f"Person {face_id}"
                if is_talking:
                    label += " - TALKING"
                
                cv2.putText(annotated, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Audio indicator
            audio_color = (0, 255, 0) if audio_active else (128, 128, 128)
            cv2.circle(annotated, (30, 30), 15, audio_color, -1)
            cv2.putText(annotated, f"Audio: {'ON' if audio_active else 'OFF'}", (50, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)
            cv2.putText(annotated, f"Vol: {audio_volume:.3f}", (50, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Stats
            with shared_state.stats_lock:
                stats = shared_state.stats.copy()
            
            info_text = f"Video Frames: {stats['video_frames']} | Audio Chunks: {stats['audio_chunks']}"
            cv2.putText(annotated, info_text, (10, annotated.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Sync info
            sync_diff = abs(latest_video_time - audio_time)
            sync_text = f"Sync diff: {sync_diff*1000:.1f}ms"
            cv2.putText(annotated, sync_text, (10, annotated.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Display transcript
            if current_transcript:
                # Draw transcript background
                transcript_y = annotated.shape[0] - 120
                transcript_height = 100
                cv2.rectangle(annotated, (10, transcript_y), 
                            (annotated.shape[1] - 10, transcript_y + transcript_height),
                            (0, 0, 0), -1)
                cv2.rectangle(annotated, (10, transcript_y), 
                            (annotated.shape[1] - 10, transcript_y + transcript_height),
                            (0, 255, 255), 2)
                
                # Split transcript into lines if too long
                max_width = annotated.shape[1] - 40
                words = current_transcript.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    (text_w, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    if text_w > max_width and current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        current_line = test_line
                if current_line:
                    lines.append(current_line)
                
                # Display transcript lines
                transcript_label = "📝 Transcript:"
                cv2.putText(annotated, transcript_label, (20, transcript_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                line_y = transcript_y + 55
                for i, line in enumerate(lines[:2]):  # Show max 2 lines
                    cv2.putText(annotated, line, (20, line_y + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show transcript history (last few)
            if transcript_history:
                history_y = 120
                cv2.putText(annotated, "Recent:", (10, history_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                for i, item in enumerate(list(transcript_history[-3:])[::-1]):  # Last 3, reversed
                    text = item['text'][:50] + "..." if len(item['text']) > 50 else item['text']
                    cv2.putText(annotated, f"• {text}", (10, history_y + 20 + i * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            cv2.imshow('Multi-threaded Video + Audio', annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                shared_state.running.clear()
                break
    
    except Exception as e:
        print(f"❌ Display thread error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()
        print("🖥️  Display thread stopped")


def main():
    """Main function to run multi-threaded video and audio analysis."""
    
    # Configuration
    VIDEO_SOURCE = 0  # Webcam, or "screen" for screen capture
    AUDIO_SAVE_FILE = "recorded_audio.wav"  # None to disable saving
    
    print("\n" + "="*60)
    print("Multi-threaded Video + Audio Analysis")
    print("="*60)
    print("Press 'q' to quit")
    print()
    
    # Create shared state
    shared_state = SharedState()
    
    # Start video thread
    video_t = threading.Thread(
        target=video_thread,
        args=(VIDEO_SOURCE, shared_state),
        daemon=True
    )
    
    # Start audio thread
    audio_t = threading.Thread(
        target=audio_thread,
        args=(shared_state, AUDIO_SAVE_FILE),
        kwargs={'enable_transcription': TRANSCRIPTION_AVAILABLE},
        daemon=True
    )
    
    # Wait for both threads to be ready
    video_t.start()
    audio_t.start()
    
    shared_state.video_ready.wait(timeout=5)
    shared_state.audio_ready.wait(timeout=5)
    
    if not shared_state.video_ready.is_set():
        print("❌ Video thread failed to start")
        shared_state.running.clear()
        return
    
    if not shared_state.audio_ready.is_set():
        print("❌ Audio thread failed to start")
        shared_state.running.clear()
        return
    
    print("✅ Both threads started successfully")
    print()
    
    # Run display in main thread
    try:
        display_thread(shared_state)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        # Stop all threads
        shared_state.running.clear()
        
        # Wait for threads to finish
        video_t.join(timeout=2)
        audio_t.join(timeout=2)
        
        print("\n✅ All threads stopped")
        if AUDIO_SAVE_FILE:
            print(f"💾 Audio saved to: {AUDIO_SAVE_FILE}")


if __name__ == "__main__":
    main()

