#!/usr/bin/env python3
"""
Test script for VideoWriter 4-channel format support (BGRA, RGBA, BGRX, RGBX)
This tests the fix for issue #28296
"""

import numpy as np
import cv2
import os
import tempfile
import sys

def test_4channel_format(format_name, test_size=(320, 240), num_frames=10):
    """
    Test writing 4-channel frames with a specific format
    
    Args:
        format_name: Format string (BGRA, RGBA, BGRX, RGBX)
        test_size: Frame size (width, height)
        num_frames: Number of frames to write
    """
    print(f"\n=== Testing {format_name} format ===")
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        output_file = tmp.name
    
    try:
        width, height = test_size
        fps = 30.0
        
        # Create a manual GStreamer pipeline for writing
        # Using a simple pipeline that accepts the format and encodes to H.264
        writer_pipeline = (
            f"appsrc ! "
            f"video/x-raw, format={format_name}, width={width}, height={height}, framerate={fps}/1 ! "
            "videoconvert ! "
            "x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "
            "mp4mux ! "
            f"filesink location={output_file}"
        )
        
        print(f"Pipeline: {writer_pipeline}")
        
        # Open VideoWriter with manual pipeline
        fourcc = 0  # Let GStreamer handle it
        writer = cv2.VideoWriter(writer_pipeline, cv2.CAP_GSTREAMER, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"ERROR: Failed to open VideoWriter for {format_name}")
            return False
        
        print(f"VideoWriter opened successfully for {format_name}")
        
        # Generate and write test frames
        frames_written = 0
        for i in range(num_frames):
            # Create a 4-channel test frame (BGRA format)
            # Fill with a gradient pattern that changes per frame
            frame = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Create a colorful gradient pattern
            for y in range(height):
                for x in range(width):
                    # B channel
                    frame[y, x, 0] = (x * 255 // width + i * 10) % 256
                    # G channel
                    frame[y, x, 1] = (y * 255 // height + i * 15) % 256
                    # R channel
                    frame[y, x, 2] = ((x + y) * 255 // (width + height) + i * 20) % 256
                    # A channel (alpha)
                    frame[y, x, 3] = 255
            
            # Verify frame shape
            assert frame.shape == (height, width, 4), f"Frame shape mismatch: {frame.shape}"
            assert frame.dtype == np.uint8, f"Frame dtype mismatch: {frame.dtype}"
            
            # Write the frame
            ret = writer.write(frame)
            if ret:
                frames_written += 1
            else:
                print(f"WARNING: Failed to write frame {i}")
        
        writer.release()
        
        print(f"Successfully wrote {frames_written}/{num_frames} frames")
        
        # Check if output file was created and has reasonable size
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"Output file created: {output_file} ({file_size} bytes)")
            if file_size > 0:
                return True
            else:
                print("ERROR: Output file is empty")
                return False
        else:
            print("ERROR: Output file was not created")
            return False
            
    except Exception as e:
        print(f"ERROR: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing VideoWriter 4-channel format support")
    print("=" * 60)
    
    # Check if GStreamer backend is available
    if not cv2.videoio_registry.hasBackend(cv2.CAP_GSTREAMER):
        print("ERROR: GStreamer backend is not available")
        print("Please ensure OpenCV was built with GStreamer support")
        return 1
    
    print("GStreamer backend is available")
    
    # Test all 4-channel formats
    formats = ["BGRA", "RGBA", "BGRX", "RGBX"]
    results = {}
    
    for fmt in formats:
        results[fmt] = test_4channel_format(fmt, test_size=(320, 240), num_frames=5)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for fmt, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{fmt:6s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

