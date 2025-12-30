#!/usr/bin/env python

'''
Robust Image Display Sample

This sample demonstrates how to display images in a loop to prevent
Python kernel freezing (common in Jupyter/Spyder) and properly handle
window close events.

Usage:
    python robust_imshow.py
'''

import cv2
import numpy as np
import time

def show_image_safe(img, winname='Image Window', time_limit_s=60):
    """
    Displays an image in an OpenCV window with a built-in event loop.
    
    Args:
        img: The image (numpy array) to display.
        winname: The name of the window.
        time_limit_s: Maximum time to keep window open (seconds).
    """
    print(f"Displaying '{winname}'. Press any key to close, or click X.")
    
    # Create window with flag to allow resizing
    cv2.namedWindow(winname, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(winname, img)
    
    start_time = time.perf_counter()
    elapsed = 0
    
    while elapsed < time_limit_s:
        # Wait for key press (100ms delay)
        # cv2.waitKey returns the ASCII value of the key pressed or -1
        key_code = cv2.waitKey(100)
        
        # Check if window was closed by user (clicking X)
        # getWindowProperty returns -1 if the window is closed
        window_status = cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE)
        
        # Break loop if a key is pressed (>0) or window is closed (<1)
        if key_code > 0 or window_status < 1:
            print("Window closed or key pressed.")
            break
            
        elapsed = time.perf_counter() - start_time
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Create a dummy image (black background with a circle)
    dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (200, 200), 100, (0, 255, 0), -1)
    
    print("Testing robust display function...")
    show_image_safe(dummy_img, "Test Window", time_limit_s=10)
    print("Done.")