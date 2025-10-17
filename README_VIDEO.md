# Video Analysis for Face Detection & Talking Detection

Quick guide to analyze video sources with computer vision to detect faces and determine who is talking.

## Installation

```bash
pip install -r requirements_video_analysis.txt
```

This installs only 3 dependencies:
- **opencv-python** - Video processing
- **mediapipe** - Face detection and facial landmarks
- **numpy** - Numerical operations

## Quick Start

### 1. Simple Demo (Recommended)

```bash
python simple_video_demo.py
```

This opens your webcam and detects faces + talking in real-time.

### 2. Full Template

```bash
python video_analysis_template.py
```

More structured code with better architecture for custom modifications.

### 3. Process a Video File

```python
from video_analysis_template import VideoAnalyzer

# Analyze video file
analyzer = VideoAnalyzer(video_source="your_video.mp4")
analyzer.run(display=True, output_path="analyzed_output.mp4")
```

## How It Works

### Face Detection
Uses MediaPipe Face Detection to identify faces in each frame.

### Talking Detection
Analyzes mouth movement by:
1. Extracting facial landmarks around the mouth
2. Calculating Mouth Aspect Ratio (MAR) = vertical_distance / horizontal_distance
3. Tracking MAR variance over time
4. High variance = mouth moving = talking

### Visual Feedback
- **Green box** = Person is talking
- **Blue/Red box** = Person is silent

## Examples

### Basic Usage

```python
from video_analysis_template import VideoAnalyzer

# Webcam
analyzer = VideoAnalyzer(video_source=0)
analyzer.run()

# Video file
analyzer = VideoAnalyzer(video_source="meeting.mp4")
analyzer.run(output_path="meeting_analyzed.mp4")
```

### Process Single Frame

```python
analyzer = VideoAnalyzer(video_source=0)
ret, frame = analyzer.cap.read()

if ret:
    annotated_frame, faces = analyzer.process_frame(frame)
    
    for face in faces:
        print(f"Face {face.id}: Talking={face.is_talking}, Confidence={face.confidence:.2f}")
```

### Adjust Sensitivity

```python
from video_analysis_template import FaceDetector, MouthMovementDetector

# More sensitive face detection
face_detector = FaceDetector(min_detection_confidence=0.5)

# More sensitive talking detection (lower threshold)
mouth_detector = MouthMovementDetector()
is_talking = mouth_detector.detect_talking(frame, face, threshold=0.03)
```

## Configuration

### Video Sources

```python
# Webcam (default camera)
VideoAnalyzer(video_source=0)

# Another camera
VideoAnalyzer(video_source=1)

# Video file
VideoAnalyzer(video_source="video.mp4")

# Video stream URL
VideoAnalyzer(video_source="rtsp://...")
```

### Save Output

```python
analyzer.run(output_path="output.mp4")
```

## Troubleshooting

### No faces detected
- Ensure good lighting
- Check if camera is working: `ls /dev/video*` (Linux) or System Preferences (Mac)
- Lower detection confidence threshold
- Make sure faces are clearly visible

### False talking detections
- Increase threshold: `detect_talking(frame, face, threshold=0.08)`
- Ensure stable camera (movement can cause false positives)

### Low FPS / Slow performance
```python
# Resize frames for faster processing
frame = cv2.resize(frame, (640, 480))

# Skip frames
if frame_count % 2 == 0:  # Process every 2nd frame
    process_frame(frame)
```

### Camera not opening
```python
# Try different indices
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera at index {i}")
        break
```

## Integration with Existing Codebase

You have the `diart` library for audio-based speaker diarization. You can combine:

```python
# Video analysis for visual detection
video_faces = video_analyzer.process_frame(frame)

# Audio analysis for speaker diarization
audio_speakers = diart_pipeline.process_audio(audio_chunk)

# Combine: Match faces with audio speakers
# This creates a multi-modal understanding of who's speaking
```

## Next Steps

1. **Face Tracking** - Track faces across frames for consistent IDs
2. **Face Recognition** - Identify specific people
3. **Audio Integration** - Combine with audio analysis from diart
4. **Emotion Detection** - Add emotion recognition
5. **Gaze Detection** - Track where people are looking

## Files

- `video_analysis_template.py` - Main template with full features
- `simple_video_demo.py` - Minimal working example
- `requirements_video_analysis.txt` - Dependencies
- `README_VIDEO.md` - This file

## Resources

- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

