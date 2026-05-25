#!/usr/bin/env python

'''
ArUco2 marker detection sample.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except IndexError:
        video_src = 0

    cap = cv.VideoCapture(video_src)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Dictionary to use
    dictionary = cv.aruco2.DICT_ARUCO_MIP_36h12

    print("Detecting markers from dictionary: DICT_ARUCO_MIP_36h12")
    print("Press 'q' to quit, 'g' to generate a marker")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect markers
        markers = cv.aruco2.detectFiducialMarkers(frame, dictionary)

        if len(markers) > 0:
            # Draw detected markers
            cv.aruco2.drawFiducialMarkers(frame, markers)
            
            for m in markers:
                print("Detected marker ID: %d" % m.id)

        cv.imshow('ArUco2 Detection', frame)
        
        ch = cv.waitKey(1) & 0xFF
        if ch == ord('q'):
            break
        elif ch == ord('g'):
            marker_img = cv.aruco2.getFiducialMarkerImage(dictionary, 42, bitSize=20)
            cv.imshow('Generated Marker 42', marker_img)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
