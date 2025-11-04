# Video Analysis Template Guide

This guide explains how to use the video analysis templates for face detection and talking detection.

## Overview

The templates provide two implementations:

1. **`video_analysis_template.py`** - Full-featured template with modular architecture
2. **`simple_video_demo.py`** - Simplified version for quick testing

## Features

- ✅ Real-time face detection
- ✅ Talking detection via mouth movement analysis
- ✅ Support for webcam and video files
- ✅ Optional video output recording
- ✅ Visual annotations (bounding boxes, labels)
- ✅ Multiple face tracking

## Installation

Install the required dependencies:

```bash
pip install -r requirements_video_analysis.txt
```

Or install manually:

```bash
pip install opencv-python mediapipe numpy pyaudio
```

## Quick Start

### Option 1: Simple Demo (Recommended for testing)

```python
python simple_video_demo.py
```

This will open your webcam and start detecting faces and talking in real-time.

### Option 2: Full Template

```python
python video_analysis_template.py
```

## Usage Examples

### 1. Analyze Webcam Feed

```python
from video_analysis_template import VideoAnalyzer

# Initialize with webcam
analyzer = VideoAnalyzer(video_source=0)

# Run analysis
analyzer.run(display=True)
```

### 2. Analyze Video File

```python
analyzer = VideoAnalyzer(video_source="path/to/video.mp4")
analyzer.run(display=True)
```

### 3. Save Output Video

```python
analyzer = VideoAnalyzer(video_source="input.mp4")
analyzer.run(display=True, output_path="output.mp4")
```

### 4. Process Single Frame

```python
analyzer = VideoAnalyzer(video_source=0)
ret, frame = analyzer.cap.read()

if ret:
    annotated_frame, faces = analyzer.process_frame(frame)

    # Check results
    for face in faces:
        print(f"Face {face.id}: Talking={face.is_talking}")
```

## Architecture

### Main Components

#### 1. FaceDetector

Detects faces in video frames using MediaPipe Face Detection.

```python
face_detector = FaceDetector(min_detection_confidence=0.7)
faces = face_detector.detect_faces(frame)
```

#### 2. MouthMovementDetector

Analyzes mouth movement to detect talking using MediaPipe Face Mesh.

```python
mouth_detector = MouthMovementDetector()
is_talking = mouth_detector.detect_talking(frame, face)
```

#### 3. AudioAnalyzer

Analyzes audio stream for speech detection (complementary to visual detection).

```python
audio_analyzer = AudioAnalyzer()
has_speech = audio_analyzer.detect_speech(audio_chunk)
```

#### 4. VideoAnalyzer

Main orchestrator that combines all components.

## Configuration

### Adjust Detection Sensitivity

```python
# Face detection confidence
face_detector = FaceDetector(min_detection_confidence=0.5)  # Lower = more sensitive

# Talking detection threshold
is_talking = mouth_detector.detect_talking(frame, face, threshold=0.03)  # Lower = more sensitive
```

### Video Output Settings

```python
# Change output codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'mp4v', 'H264'
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
```

## How It Works

### Face Detection

Uses MediaPipe Face Detection with two models:

- Model 0: Short-range (within 2 meters)
- Model 1: Full-range (within 5 meters) - default

### Talking Detection

The system detects talking through **Mouth Aspect Ratio (MAR)**:

1. Extracts facial landmarks around the mouth
2. Calculates vertical distance (upper lip to lower lip)
3. Calculates horizontal distance (left corner to right corner)
4. Computes MAR = vertical / horizontal
5. Tracks variance in MAR over time
6. High variance indicates mouth movement (talking)

```
MAR Variance > threshold → Talking detected
```

### Visual Indicators

- **Green box** = Person is talking
- **Blue/Red box** = Person is silent
- **Confidence score** = Detection confidence (0.0 - 1.0)

## Advanced Usage

### Custom Face Tracking

```python
class CustomAnalyzer(VideoAnalyzer):
    def process_frame(self, frame):
        annotated_frame, faces = super().process_frame(frame)

        # Add custom logic
        for face in faces:
            if face.is_talking:
                # Do something when talking detected
                self.log_talking_event(face)

        return annotated_frame, faces
```

### Integrate with Audio Analysis

```python
import wave
import numpy as np

# Read audio from video
audio_data = extract_audio_from_video("video.mp4")

# Analyze synchronously
while analyzing:
    frame = get_next_frame()
    audio_chunk = get_next_audio_chunk()

    faces = face_detector.detect_faces(frame)
    has_speech = audio_analyzer.detect_speech(audio_chunk)

    # Combine visual and audio cues
    for face in faces:
        face.is_talking = (
            mouth_detector.detect_talking(frame, face) and
            has_speech
        )
```

### Performance Optimization

```python
# Reduce frame processing rate
frame_skip = 2
for i, frame in enumerate(video_frames):
    if i % frame_skip != 0:
        continue
    process_frame(frame)

# Resize frames for faster processing
frame = cv2.resize(frame, (640, 480))

# Use GPU acceleration (if available)
# MediaPipe automatically uses GPU when available
```

## Troubleshooting

### No faces detected

- Ensure good lighting
- Check camera is working
- Lower `min_detection_confidence` threshold
- Ensure faces are visible and not too far

### False positive talking detection

- Increase `threshold` in `detect_talking()` (try 0.08-0.1)
- Increase mouth history window size
- Add audio confirmation

### Low FPS

- Reduce video resolution
- Skip frames (process every 2nd or 3rd frame)
- Use simpler detection model
- Close other applications

### Camera not opening

```python
# Try different camera indices
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        break
```

## Integration with Diart (Speaker Diarization)

The project includes `diart` for audio-based speaker diarization. You can combine it with face detection:

```python
from diart import SpeakerDiarization
from video_analysis_template import VideoAnalyzer

# Initialize both
video_analyzer = VideoAnalyzer(video_source="video.mp4")
audio_diarization = SpeakerDiarization()

# Process video + audio
# Match detected faces with speaker segments
# Create comprehensive talking timeline
```

## Next Steps

1. **Add Face Recognition**: Identify specific people using face embeddings
2. **Track Faces**: Implement tracking across frames for consistent IDs
3. **Audio Sync**: Synchronize with audio-based speech detection
4. **Emotion Detection**: Add emotion recognition from facial expressions
5. **Gaze Detection**: Track where people are looking
6. **Multi-person Conversations**: Analyze turn-taking patterns

## Resources

- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Diart Speaker Diarization](https://github.com/juanmc2005/StreamingSpeakerDiarization)

## License

This template is provided as-is for use in the CocktailPartyAI project.
