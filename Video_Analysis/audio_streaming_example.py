"""
How to Get Audio Chunks from a Stream in Real-Time

This demonstrates different approaches to get audio chunks from PyAudio stream.
"""

import pyaudio
import numpy as np
import time
from queue import Queue


# ============================================================================
# METHOD 1: BLOCKING READS (Simplest - Current implementation)
# ============================================================================

def method1_blocking_reads():
    """
    BLOCKING approach: stream.read() waits until chunk is ready.
    
    This is the simplest and most common approach.
    When you call stream.read(chunk_size), it blocks until that many samples
    are available, then returns them. In a loop, this automatically gives you
    real-time chunks.
    """
    print("\n=== METHOD 1: Blocking Reads ===")
    
    # Setup
    chunk_size = 1024  # samples per chunk
    sample_rate = 16000
    format = pyaudio.paInt16
    channels = 1
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )
    
    stream.start_stream()
    
    try:
        chunk_count = 0
        start_time = time.time()
        
        while chunk_count < 100:  # Process 100 chunks
            # BLOCKING READ: This waits until chunk_size samples are available
            # When you loop this, you automatically get chunks in real-time!
            audio_data = stream.read(chunk_size, exception_on_overflow=False)
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Process chunk
            rms = np.sqrt(np.mean(audio_array**2))
            chunk_count += 1
            
            if chunk_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Chunk {chunk_count}: RMS={rms:.4f}, Rate={chunk_count/elapsed:.1f} chunks/sec")
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


# ============================================================================
# METHOD 2: NON-BLOCKING READS (Check availability first)
# ============================================================================

def method2_non_blocking_reads():
    """
    NON-BLOCKING approach: Check if data is available first.
    
    Use this if you don't want to block and have other things to do.
    """
    print("\n=== METHOD 2: Non-Blocking Reads ===")
    
    chunk_size = 1024
    sample_rate = 16000
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )
    
    stream.start_stream()
    
    try:
        chunk_count = 0
        start_time = time.time()
        
        while chunk_count < 100:
            # Check how many frames are available
            frames_available = stream.get_read_available()
            
            if frames_available >= chunk_size:
                # We have enough data, read it (won't block)
                audio_data = stream.read(chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                rms = np.sqrt(np.mean(audio_array**2))
                chunk_count += 1
                
                if chunk_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Chunk {chunk_count}: RMS={rms:.4f}")
            else:
                # Not enough data yet, do other work or sleep
                time.sleep(0.001)  # Small sleep to avoid busy-waiting
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


# ============================================================================
# METHOD 3: CALLBACK-BASED (True real-time streaming)
# ============================================================================

def method3_callback_based():
    """
    CALLBACK approach: PyAudio calls your function when data is ready.
    
    This is the most efficient for real-time processing. PyAudio automatically
    calls your callback function whenever a chunk is ready, so you don't need
    to poll or block.
    """
    print("\n=== METHOD 3: Callback-Based Streaming ===")
    
    chunk_size = 1024
    sample_rate = 16000
    audio_queue = Queue()  # Thread-safe queue to pass chunks
    
    def audio_callback(in_data, frame_count, time_info, status):
        """
        This function is called automatically by PyAudio when data is ready.
        
        Args:
            in_data: Raw audio bytes
            frame_count: Number of frames (should equal chunk_size)
            time_info: Timing information
            status: Status flags
        
        Returns:
            (None, pyaudio.paContinue) to keep streaming
        """
        # Convert to numpy array
        audio_array = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Process and put in queue (non-blocking)
        rms = np.sqrt(np.mean(audio_array**2))
        try:
            audio_queue.put_nowait({
                'data': audio_array,
                'rms': rms,
                'timestamp': time.time()
            })
        except:
            pass  # Queue full, drop chunk
        
        # Return None and continue flag to keep streaming
        return (None, pyaudio.paContinue)
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
        stream_callback=audio_callback  # Set the callback!
    )
    
    stream.start_stream()
    
    try:
        chunk_count = 0
        start_time = time.time()
        
        # Main loop - get chunks from queue (produced by callback)
        while chunk_count < 100:
            try:
                chunk = audio_queue.get(timeout=0.1)
                chunk_count += 1
                
                if chunk_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Chunk {chunk_count}: RMS={chunk['rms']:.4f}, "
                          f"Rate={chunk_count/elapsed:.1f} chunks/sec")
            except:
                continue
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


# ============================================================================
# METHOD 4: Using in a Thread (Like your current implementation)
# ============================================================================

import threading

def method4_threaded_streaming():
    """
    THREADED approach: Run audio reading in a separate thread.
    
    This is what your current implementation does - keeps the main thread
    free while audio processing happens in background.
    """
    print("\n=== METHOD 4: Threaded Streaming ===")
    
    chunk_size = 1024
    sample_rate = 16000
    audio_queue = Queue()
    running = threading.Event()
    running.set()
    
    def audio_thread():
        """Thread that continuously reads audio chunks."""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        stream.start_stream()
        
        chunk_count = 0
        try:
            while running.is_set():
                # Blocking read - waits for chunk
                audio_data = stream.read(chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                rms = np.sqrt(np.mean(audio_array**2))
                
                # Put in queue for main thread
                audio_queue.put({
                    'data': audio_array,
                    'rms': rms,
                    'chunk_num': chunk_count,
                    'timestamp': time.time()
                })
                
                chunk_count += 1
                
                if chunk_count >= 100:
                    break
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
    
    # Start audio thread
    thread = threading.Thread(target=audio_thread, daemon=True)
    thread.start()
    
    # Main thread - process chunks from queue
    try:
        processed = 0
        start_time = time.time()
        
        while processed < 100:
            try:
                chunk = audio_queue.get(timeout=0.5)
                processed += 1
                
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {processed}: RMS={chunk['rms']:.4f}, "
                          f"Rate={processed/elapsed:.1f} chunks/sec")
            except:
                if not thread.is_alive():
                    break
    
    finally:
        running.clear()
        thread.join(timeout=2)


# ============================================================================
# EXPLANATION: How Real-Time Chunking Works
# ============================================================================

"""
HOW AUDIO STREAMING WORKS IN REAL-TIME:

1. **Buffer Size (chunk_size)**:
   - You specify how many samples you want per chunk (e.g., 1024 samples)
   - At 16kHz sample rate: 1024 samples = 1024/16000 = 0.064 seconds of audio
   - Smaller chunks = lower latency, but more processing overhead
   - Larger chunks = higher latency, but less overhead

2. **Blocking Reads (Method 1)**:
   - stream.read(chunk_size) BLOCKS until chunk_size samples are captured
   - In a loop: automatically waits for each chunk → real-time streaming!
   - Time between chunks ≈ chunk_size / sample_rate seconds
   - Example: 1024 samples / 16000 Hz = 64ms per chunk

3. **Non-Blocking Reads (Method 2)**:
   - Check stream.get_read_available() first
   - Only read if enough data is available
   - Good for when you have other work to do

4. **Callback-Based (Method 3)**:
   - PyAudio calls your function automatically when chunk is ready
   - Most efficient - no polling needed
   - Callback runs in separate thread managed by PyAudio

5. **Threaded (Method 4)**:
   - Run audio reading in separate thread
   - Main thread processes chunks from queue
   - Good for complex applications with multiple responsibilities

WHICH METHOD TO USE:
- **Simple application**: Method 1 (blocking reads)
- **Need to do other work**: Method 2 (non-blocking) or Method 4 (threaded)
- **Maximum efficiency**: Method 3 (callback-based)
- **Complex multi-threaded app**: Method 4 (threaded with queue)

YOUR CURRENT IMPLEMENTATION uses Method 4 (threaded with blocking reads),
which is perfect for your use case!
"""

if __name__ == "__main__":
    print("=" * 70)
    print("Audio Streaming Methods Demo")
    print("=" * 70)
    print("\n⚠️  Note: These will actually record from your microphone!")
    print("Press Ctrl+C to stop each method early.\n")
    
    try:
        # Uncomment the method you want to test:
        # method1_blocking_reads()
        # method2_non_blocking_reads()
        # method3_callback_based()
        # method4_threaded_streaming()
        
        print("\n✅ Demo complete!")
        print("\nTo test a method, uncomment it in the code and run this script.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")


