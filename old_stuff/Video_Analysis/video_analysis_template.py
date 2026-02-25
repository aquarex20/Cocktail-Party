"""
Video Analysis Template: Face Detection + Speech Detection
This template processes video sources to detect faces and determine who is talking.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import wave
import pyaudio
from threading import Thread
import time


@dataclass
class Face:
    """Represents a detected face in the video."""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]
    is_talking: bool = False
    confidence: float = 0.0
    landmarks: Optional[np.ndarray] = None


class FaceDetector:
    """Handles face detection using MediaPipe Face Detection."""
    
    def __init__(self, min_detection_confidence=0.7):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=1  # 1 for full range detection, 0 for short range
        )
    
    def detect_faces(self, frame: np.ndarray) -> List[Face]:
        """
        Detect faces in the given frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of detected Face objects
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for idx, detection in enumerate(results.detections):
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Calculate center
                center_x = x + width // 2
                center_y = y + height // 2
                
                face = Face(
                    id=idx,
                    bbox=(x, y, width, height),
                    center=(center_x, center_y),
                    confidence=detection.score[0]
                )
                faces.append(face)
        
        return faces
    
    def __del__(self):
        self.face_detection.close()


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
        # Mouth landmark indices
        self.MOUTH_INDICES = [61, 291, 0, 17, 78, 308]
        self.mouth_history = {}  # Store mouth aspect ratios per face
        
    def calculate_mouth_aspect_ratio(self, landmarks, frame_shape) -> float:
        """Calculate mouth aspect ratio (MAR) to detect talking."""
        h, w = frame_shape[:2]
        
        # Get key mouth points
        # Upper lip center: 13, Lower lip center: 14
        # Left corner: 61, Right corner: 291
        upper = landmarks[13]
        lower = landmarks[14]
        left = landmarks[61]
        right = landmarks[291]
        
        # Convert to pixel coordinates
        upper_y = upper.y * h
        lower_y = lower.y * h
        left_x = left.x * w
        right_x = right.x * w
        
        # Calculate vertical and horizontal distances
        vertical = abs(lower_y - upper_y)
        horizontal = abs(right_x - left_x)
        
        # Mouth aspect ratio
        mar = vertical / (horizontal + 1e-6)
        return mar
    
    def detect_talking(self, frame: np.ndarray, face: Face, threshold=0.05) -> bool:
        """
        Detect if a person is talking based on mouth movement.
        
        Args:
            frame: BGR image
            face: Face object with bounding box
            threshold: Threshold for mouth movement variance
            
        Returns:
            True if talking detected
        """
        x, y, w, h = face.bbox
        
        # Expand ROI slightly
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return False
        
        # Convert to RGB
        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_roi)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            mar = self.calculate_mouth_aspect_ratio(landmarks, face_roi.shape)
            
            # Initialize history for this face
            if face.id not in self.mouth_history:
                self.mouth_history[face.id] = deque(maxlen=10)
            
            self.mouth_history[face.id].append(mar)
            
            # Check variance in mouth aspect ratio
            if len(self.mouth_history[face.id]) >= 5:
                variance = np.var(list(self.mouth_history[face.id]))
                return variance > threshold
        
        return False
    
    def __del__(self):
        self.face_mesh.close()


class AudioAnalyzer:
    """Analyzes audio to detect speech activity."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_buffer = deque(maxlen=50)
        self.is_speech_active = False
        
    def calculate_audio_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate energy of audio chunk."""
        return np.sum(audio_chunk ** 2) / len(audio_chunk)
    
    def detect_speech(self, audio_chunk: np.ndarray, threshold=0.01) -> bool:
        """
        Simple speech detection based on audio energy.
        
        Args:
            audio_chunk: Audio samples
            threshold: Energy threshold for speech
            
        Returns:
            True if speech detected
        """
        energy = self.calculate_audio_energy(audio_chunk)
        self.audio_buffer.append(energy)
        
        if len(self.audio_buffer) >= 10:
            avg_energy = np.mean(list(self.audio_buffer))
            return avg_energy > threshold
        
        return False


class VideoAnalyzer:
    """Main class for video analysis combining face and speech detection."""
    
    def __init__(self, video_source=0):
        """
        Initialize video analyzer.
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
        """
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Initialize detectors
        self.face_detector = FaceDetector()
        self.mouth_detector = MouthMovementDetector()
        self.audio_analyzer = AudioAnalyzer()
        
        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Face]]:
        """
        Process a single frame to detect faces and talking.
        
        Args:
            frame: Input frame
            
        Returns:
            Annotated frame and list of detected faces
        """
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        # Check for talking using mouth movement
        for face in faces:
            face.is_talking = self.mouth_detector.detect_talking(frame, face)
        
        # Annotate frame
        annotated_frame = self.annotate_frame(frame, faces)
        
        return annotated_frame, faces
    
    def annotate_frame(self, frame: np.ndarray, faces: List[Face]) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        annotated = frame.copy()
        
        for face in faces:
            x, y, w, h = face.bbox
            
            # Choose color based on talking status
            color = (0, 255, 0) if face.is_talking else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"Face {face.id}"
            if face.is_talking:
                label += " - TALKING"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - text_h - 10), (x + text_w, y), color, -1)
            
            # Text
            cv2.putText(annotated, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence
            conf_text = f"{face.confidence:.2f}"
            cv2.putText(annotated, conf_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add frame info
        info_text = f"Faces: {len(faces)} | Talking: {sum(f.is_talking for f in faces)}"
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def run(self, display=True, output_path=None):
        """
        Run the video analysis pipeline.
        
        Args:
            display: Whether to display the video
            output_path: Optional path to save output video
        """
        # Initialize video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                (self.frame_width, self.frame_height))
        
        print("Starting video analysis... Press 'q' to quit")
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video or cannot read frame")
                    break
                
                frame_count += 1
                
                # Process frame
                annotated_frame, faces = self.process_frame(frame)
                
                # Write output
                if out:
                    out.write(annotated_frame)
                
                # Display
                if display:
                    cv2.imshow('Video Analysis', annotated_frame)
                    
                    # Quit on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print stats every 30 frames
                if frame_count % 30 == 0:
                    talking_count = sum(f.is_talking for f in faces)
                    print(f"Frame {frame_count}: {len(faces)} faces, {talking_count} talking")
        
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user")
        
        finally:
            self.cleanup(out)
            print(f"Processed {frame_count} frames")
    
    def cleanup(self, out=None):
        """Release resources."""
        self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


def main():
    """Example usage of the video analyzer."""
    
    # Configuration
    VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
    DISPLAY = True
    OUTPUT_PATH = None  # Set to a path like "output.mp4" to save video
    
    # Example with video file:
    # VIDEO_SOURCE = "path/to/your/video.mp4"
    # OUTPUT_PATH = "analyzed_output.mp4"
    
    try:
        # Initialize analyzer
        analyzer = VideoAnalyzer(video_source=VIDEO_SOURCE)
        
        # Run analysis
        analyzer.run(display=DISPLAY, output_path=OUTPUT_PATH)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

