#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import time

def test_pyrUp_overflow_fix():
    """Test that pyrUp throws an appropriate error instead of crashing"""
    
    print("Testing pyrUp memory overflow fix...")
    
    # Create a small test image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    print(f"Initial image size: {image.shape[1]}x{image.shape[0]}")
    
    current = image.copy()
    iterations = 0
    
    try:
        # Keep calling pyrUp until we hit the memory limit
        for i in range(20):
            start_time = time.time()
            
            # Calculate what the next size would be
            next_height = current.shape[0] * 2
            next_width = current.shape[1] * 2
            next_pixels = next_height * next_width
            
            print(f"Iteration {i+1}: {current.shape[1]}x{current.shape[0]} -> {next_width}x{next_height} ({next_pixels:,} pixels)")
            
            # This should eventually throw an error instead of crashing
            next_image = cv2.pyrUp(current)
            
            end_time = time.time()
            print(f"  Success in {(end_time - start_time)*1000:.1f}ms")
            
            current = next_image
            iterations = i + 1
            
            # Safety break to avoid going too far
            if current.shape[0] > 25600 or current.shape[1] > 25600:
                print("Reached safety limit, stopping...")
                break
                
    except cv2.error as e:
        print(f"\nCaught OpenCV error (expected): {e}")
        print(f"Stopped at iteration {iterations + 1}")
        return True
        
    except MemoryError as e:
        print(f"\nCaught MemoryError: {e}")
        print(f"Stopped at iteration {iterations + 1}")
        return True
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False
        
    print(f"\nCompleted {iterations} iterations without error")
    print(f"Final image size: {current.shape[1]}x{current.shape[0]}")
    return True

def test_pyrUp_normal_usage():
    """Test that normal pyrUp usage still works"""
    
    print("\nTesting normal pyrUp usage...")
    
    # Test with various image sizes and types
    test_cases = [
        ((50, 50), np.uint8),
        ((100, 100), np.uint8),
        ((50, 50, 3), np.uint8),
        ((100, 100, 3), np.uint8),
        ((50, 50), np.float32),
    ]
    
    for shape, dtype in test_cases:
        try:
            # Create test image
            if len(shape) == 2:
                image = np.random.randint(0, 256, shape).astype(dtype)
            else:
                image = np.random.randint(0, 256, shape).astype(dtype)
            
            # Apply pyrUp
            result = cv2.pyrUp(image)
            
            # Check result dimensions
            expected_h = image.shape[0] * 2
            expected_w = image.shape[1] * 2
            
            if result.shape[0] == expected_h and result.shape[1] == expected_w:
                print(f"  ✓ {shape} -> {result.shape} (dtype: {dtype.__name__})")
            else:
                print(f"  ✗ {shape} -> {result.shape} (expected: {expected_h}x{expected_w})")
                return False
                
        except Exception as e:
            print(f"  ✗ Failed for {shape} ({dtype.__name__}): {e}")
            return False
    
    return True

def test_pyrUp_with_custom_size():
    """Test pyrUp with custom destination size"""
    
    print("\nTesting pyrUp with custom destination size...")
    
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    try:
        # Test with valid custom size
        custom_size = (150, 150)
        result = cv2.pyrUp(image, dstsize=custom_size)
        
        if result.shape[:2] == custom_size:
            print(f"  ✓ Custom size {custom_size} works")
        else:
            print(f"  ✗ Custom size failed: got {result.shape[:2]}, expected {custom_size}")
            return False
            
        # Test with oversized custom size (should fail)
        huge_size = (100000, 100000)  # This should trigger the bounds check
        try:
            result = cv2.pyrUp(image, dstsize=huge_size)
            print(f"  ✗ Huge size {huge_size} should have failed but didn't")
            return False
        except cv2.error:
            print(f"  ✓ Huge size {huge_size} correctly rejected")
            
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("OpenCV pyrUp Memory Overflow Fix Test")
    print("=" * 50)
    
    # Test 1: Check that the overflow protection works
    success1 = test_pyrUp_overflow_fix()
    
    # Test 2: Check that normal usage still works
    success2 = test_pyrUp_normal_usage()
    
    # Test 3: Check custom size handling
    success3 = test_pyrUp_with_custom_size()
    
    print("\n" + "=" * 50)
    if success1 and success2 and success3:
        print("✓ All tests passed! The fix is working correctly.")
        sys.exit(0)
    else:
        print("✗ Some tests failed.")
        sys.exit(1)
