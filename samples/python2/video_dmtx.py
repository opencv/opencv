#!/usr/bin/env python

'''
Data matrix detector sample.
Usage:
   video_dmtx {<video device number>|<video file name>}

   Generate a datamatrix from  from http://datamatrix.kaywa.com/ and print it out.
   NOTE: This only handles data matrices, generated for text strings of max 3 characters

   Resize the screen to be large enough for your camera to see, and it should find an read it.

Keyboard shortcuts:

   q or ESC - exit
   space - save current image as datamatrix<frame_number>.jpg
'''

import cv2
import numpy as np
import sys

def data_matrix_demo(cap):
    window_name = "Data Matrix Detector"
    frame_number = 0
    need_to_save = False

    while 1:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        codes, corners, dmtx = cv2.findDataMatrix(gray)

        cv2.drawDataMatrixCodes(frame, codes, corners)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(30)
        c = chr(key & 255)
        if c in ['q', 'Q', chr(27)]:
            break

        if c == ' ':
            need_to_save = True

        if need_to_save and codes:
            filename = ("datamatrix%03d.jpg" % frame_number)
            cv2.imwrite(filename, frame)
            print "Saved frame to " + filename
            need_to_save = False

        frame_number += 1


if __name__ == '__main__':
    print __doc__

    if len(sys.argv) == 1:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(sys.argv[1])
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(sys.argv[1]))

    if not cap.isOpened():
        print 'Cannot initialize video capture'
        sys.exit(-1)

    data_matrix_demo(cap)
