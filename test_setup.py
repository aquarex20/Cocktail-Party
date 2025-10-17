"""
Test script to verify video analysis setup.
Run this to check if all dependencies are correctly installed.
"""

import sys


def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    tests = {
        'OpenCV': 'cv2',
        'MediaPipe': 'mediapipe',
        'NumPy': 'numpy',
        'PyAudio': 'pyaudio'
    }
    
    results = {}
    
    for name, module in tests.items():
        try:
            __import__(module)
            results[name] = '✓ OK'
            print(f"  {name}: ✓ OK")
        except ImportError as e:
            results[name] = f'✗ FAILED: {str(e)}'
            print(f"  {name}: ✗ FAILED")
    
    return all('OK' in v for v in results.values())


def test_camera():
    """Test if camera is accessible."""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("  Camera: ✗ FAILED - Cannot open camera")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"  Camera: ✓ OK - Resolution: {w}x{h}")
            return True
        else:
            print("  Camera: ✗ FAILED - Cannot read frame")
            return False
            
    except Exception as e:
        print(f"  Camera: ✗ FAILED - {str(e)}")
        return False


def test_mediapipe():
    """Test MediaPipe face detection."""
    print("\nTesting MediaPipe face detection...")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = face_detection.process(dummy_image)
        
        face_detection.close()
        print("  MediaPipe Face Detection: ✓ OK")
        return True
        
    except Exception as e:
        print(f"  MediaPipe Face Detection: ✗ FAILED - {str(e)}")
        return False


def test_versions():
    """Print versions of installed packages."""
    print("\nInstalled versions:")
    
    try:
        import cv2
        print(f"  OpenCV: {cv2.__version__}")
    except:
        pass
    
    try:
        import mediapipe as mp
        print(f"  MediaPipe: {mp.__version__}")
    except:
        pass
    
    try:
        import numpy as np
        print(f"  NumPy: {np.__version__}")
    except:
        pass
    
    try:
        import pyaudio
        print(f"  PyAudio: {pyaudio.__version__}")
    except:
        pass


def main():
    """Run all tests."""
    print("=" * 50)
    print("Video Analysis Setup Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Some required packages are missing!")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements_video_analysis.txt")
        sys.exit(1)
    
    # Test versions
    test_versions()
    
    # Test camera (optional)
    camera_ok = test_camera()
    
    # Test MediaPipe
    mediapipe_ok = test_mediapipe()
    
    # Summary
    print("\n" + "=" * 50)
    if imports_ok and mediapipe_ok:
        print("✓ All tests passed!")
        if not camera_ok:
            print("\n⚠️  Camera test failed, but you can still process video files.")
        print("\nYou can now run:")
        print("  python simple_video_demo.py")
        print("  python video_analysis_template.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
    
    print("=" * 50)


if __name__ == "__main__":
    main()

