#!/usr/bin/env python

'''
    This program demonstrates Laplace point/edge detection using
    OpenCV function Laplacian()
    It captures from the camera of your choice: 0, 1, ... default 0
    Usage:
        python laplace.py <ddepth> <smoothType> <sigma>
        If no arguments given default arguments will be used.

    Keyboard Shortcuts:
    Press space bar to exit the program.
    '''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys

def main():
    # Declare the variables we are going to use
    ddepth = cv.CV_16S
    smoothType = "MedianBlur"
    sigma = 3
    if len(sys.argv)==4:
        ddepth = sys.argv[1]
        smoothType = sys.argv[2]
        sigma = sys.argv[3]
    # Taking input from the camera
    cap=cv.VideoCapture(0)
    # Create Window and Trackbar
    cv.namedWindow("Laplace of Image", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("Kernel Size Bar", "Laplace of Image", sigma, 15, lambda x:x)
    # Printing frame width, height and FPS
    print("=="*40)
    print("Frame Width: ", cap.get(cv.CAP_PROP_FRAME_WIDTH), "Frame Height: ", cap.get(cv.CAP_PROP_FRAME_HEIGHT), "FPS: ", cap.get(cv.CAP_PROP_FPS))
    while True:
        # Reading input from the camera
        ret, frame = cap.read()
        if ret == False:
            print("Can't open camera/video stream")
            break
        # Taking input/position from the trackbar
        sigma = cv.getTrackbarPos("Kernel Size Bar", "Laplace of Image")
        # Setting kernel size
        ksize = (sigma*5)|1
        # Removing noise by blurring with a filter
        if smoothType == "GAUSSIAN":
            smoothed = cv.GaussianBlur(frame, (ksize, ksize), sigma, sigma)
        if smoothType == "BLUR":
            smoothed = cv.blur(frame, (ksize, ksize))
        if smoothType == "MedianBlur":
            smoothed = cv.medianBlur(frame, ksize)

        # Apply Laplace function
        laplace = cv.Laplacian(smoothed, ddepth, 5)
        # Converting back to uint8
        result = cv.convertScaleAbs(laplace, (sigma+1)*0.25)
        # Display Output
        cv.imshow("Laplace of Image", result)
        k = cv.waitKey(30)
        if k == 27:
            return
if __name__ == "__main__":
    print(__doc__)
    main()
    cv.destroyAllWindows()
