"""
Simplified Video Analysis Demo
A minimal example for quick testing of face and talking detection.

Uses the correct MAR (Mouth Aspect Ratio) formula with proper landmarks.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque


class SimpleVideoAnalyzer:
    """Minimal implementation for face detection and talking detection."""

    def __init__(self):
        # Face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.7,
            model_selection=1
        )

        # Face mesh for mouth detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Store mouth movement history
        self.mouth_history = deque(maxlen=15)
        self.frame_count = 0

        # Calibration
        self.baseline_mar = None
        self.calibration_values = []

        # Talking state tracking
        self.talking_frames = 0  # Count of consecutive talking frames
        self.silent_frames = 0   # Count of consecutive silent frames
        self.talking_state = False  # Current state

    def _euclidean_distance(self, p1, p2):
        """Calculate distance between two landmarks."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def calculate_mar(self, landmarks):
        """
        Calculate MAR using the standard formula.
        MAR = (N1 + N2 + N3) / (3 * D)

        Landmarks:
        - D (horizontal): 61 <-> 291 (mouth corners)
        - N1: 39 <-> 181
        - N2: 0 <-> 17 (center)
        - N3: 269 <-> 405
        """
        # Horizontal (mouth width)
        D = self._euclidean_distance(landmarks[61], landmarks[291])

        # Three vertical measurements
        N1 = self._euclidean_distance(landmarks[39], landmarks[181])
        N2 = self._euclidean_distance(landmarks[0], landmarks[17])
        N3 = self._euclidean_distance(landmarks[269], landmarks[405])

        if D < 1e-6:
            return 0.0

        MAR = (N1 + N2 + N3) / (3.0 * D)
        return MAR

    def calculate_inner_lip_distance(self, landmarks, frame_height):
        """Simple pixel distance between inner lips."""
        upper = landmarks[13]
        lower = landmarks[14]
        return abs(lower.y - upper.y) * frame_height

    def is_talking(self, mar, pixel_dist):
        """
        Detect talking using multiple methods.
        Returns (is_talking, debug_info)
        """
        self.mouth_history.append(mar)
        history = list(self.mouth_history)

        # Calibration: first 20 frames
        if self.baseline_mar is None:
            self.calibration_values.append(mar)
            if len(self.calibration_values) >= 20:
                self.baseline_mar = np.percentile(self.calibration_values, 25)  # 25th percentile = closed mouth
                print(f"[CALIBRATED] Baseline MAR: {self.baseline_mar:.3f}")
            return False, "Calibrating..."

        # Count how many detection methods trigger
        triggers = 0
        reason = ""

        # Method 1: MAR significantly above baseline
        if mar > self.baseline_mar * 1.15:  # 15% above baseline
            triggers += 1
            reason = "MAR>baseline"

        # Method 2: Absolute MAR threshold
        if mar > 0.80:  # Raised threshold
            triggers += 1
            reason = "MAR>0.80"

        # Method 3: Pixel distance (only if significant)
        if pixel_dist > 12:  # Raised threshold
            triggers += 1
            reason = "pixels>12"

        # Method 4: Movement detection (variance) - key indicator
        if len(history) >= 5:
            variance = np.var(history[-5:])
            if variance > 0.002:  # Need real movement
                triggers += 2  # Weight this higher
                reason = f"var={variance:.4f}"

        # Method 5: Recent change - needs to be significant
        if len(history) >= 2:
            change = abs(history[-1] - history[-2])
            if change > 0.05:  # Raised threshold
                triggers += 1
                reason = f"change={change:.3f}"

        # Method 6: Range in recent history
        if len(history) >= 5:
            mar_range = max(history[-5:]) - min(history[-5:])
            if mar_range > 0.08:  # Raised threshold
                triggers += 1
                reason = f"range={mar_range:.3f}"

        # Determine if currently showing talking signals
        frame_is_talking = triggers >= 2  # Need at least 2 triggers

        # State machine: require consistent signals to change state
        if frame_is_talking:
            self.talking_frames += 1
            self.silent_frames = 0
        else:
            self.silent_frames += 1
            self.talking_frames = 0

        # Change to TALKING: need 3 consecutive talking frames
        if not self.talking_state and self.talking_frames >= 3:
            self.talking_state = True

        # Change to SILENT: need 8 consecutive silent frames (slower to stop)
        if self.talking_state and self.silent_frames >= 8:
            self.talking_state = False

        return self.talking_state, reason

    def process_frame(self, frame):
        """Process frame and return annotated result."""
        self.frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Detect faces
        face_results = self.face_detection.process(rgb_frame)
        mesh_results = self.face_mesh.process(rgb_frame)

        talking = False
        mar = 0
        pixel_dist = 0
        reason = ""

        # Draw faces
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Check for talking
                if mesh_results.multi_face_landmarks:
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark

                        # Calculate MAR
                        mar = self.calculate_mar(landmarks)
                        pixel_dist = self.calculate_inner_lip_distance(landmarks, h)

                        # Check if talking
                        talking, reason = self.is_talking(mar, pixel_dist)

                # Draw box
                color = (0, 255, 0) if talking else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

                # Label
                label = "TALKING" if talking else "Silent"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show debug info on screen
        baseline_str = f"{self.baseline_mar:.3f}" if self.baseline_mar else "calibrating"
        debug_text = f"MAR: {mar:.3f} | Pixels: {pixel_dist:.1f} | Baseline: {baseline_str}"
        cv2.putText(frame, debug_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if reason:
            cv2.putText(frame, f"Trigger: {reason}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Print debug to terminal every 15 frames
        if self.frame_count % 15 == 0:
            history = list(self.mouth_history)
            variance = np.var(history[-5:]) if len(history) >= 5 else 0
            mar_range = max(history[-5:]) - min(history[-5:]) if len(history) >= 5 else 0
            print(f"[DEBUG] MAR={mar:.3f} | Pixels={pixel_dist:.1f} | Var={variance:.5f} | Range={mar_range:.3f} | Talking={talking}")

        return frame

    def run(self, source=0):
        """Run the analyzer on video source."""
        cap = cv2.VideoCapture(source)

        print("="*60)
        print("Starting video analysis... Press 'q' to quit")
        print("Keep your mouth CLOSED for the first 2 seconds to calibrate")
        print("="*60)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process
            result = self.process_frame(frame)

            # Display
            cv2.imshow('Simple Video Analysis', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    """Run the simple demo."""
    analyzer = SimpleVideoAnalyzer()

    # Use webcam (0) or provide video path
    analyzer.run(source=0)

    # For video file:
    # analyzer.run(source="your_video.mp4")


if __name__ == "__main__":
    main()

