#!/usr/bin/env python
'''
mouse_and_match.py [-i path | --input path: default ./]

Demonstrate using a mouse to interact with an image:
 Read in the images in a directory one by one
 Allow the user to select parts of an image with a mouse
 When they let go of the mouse, it correlates (using matchTemplate) that patch with the image.
 ESC to exit
'''
import numpy as np
import cv2

# built-in modules
import os
import sys
import glob
import argparse
from math import *


drag_start = None
sel = (0,0,0,0)

def onmouse(event, x, y, flags, param):
    global drag_start, sel
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = x, y
        sel = 0,0,0,0
    elif event == cv2.EVENT_LBUTTONUP:
        if sel[2] > sel[0] and sel[3] > sel[1]:
            patch = gray[sel[1]:sel[3],sel[0]:sel[2]]
            result = cv2.matchTemplate(gray,patch,cv2.TM_CCOEFF_NORMED)
            result = np.abs(result)**3
            val, result = cv2.threshold(result, 0.01, 0, cv2.THRESH_TOZERO)
            result8 = cv2.normalize(result,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            cv2.imshow("result", result8)
        drag_start = None
    elif drag_start:
        #print flags
        if flags & cv2.EVENT_FLAG_LBUTTON:
            minpos = min(drag_start[0], x), min(drag_start[1], y)
            maxpos = max(drag_start[0], x), max(drag_start[1], y)
            sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img, (sel[0], sel[1]), (sel[2], sel[3]), (0,255,255), 1)
            cv2.imshow("gray", img)
        else:
            print "selection is complete"
            drag_start = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate mouse interaction with images')
    parser.add_argument("-i","--input", default='./', help="Input directory.")
    args = parser.parse_args()
    path = args.input

    cv2.namedWindow("gray",1)
    cv2.setMouseCallback("gray", onmouse)
    '''Loop through all the images in the directory'''
    for infile in glob.glob( os.path.join(path, '*.*') ):
        ext = os.path.splitext(infile)[1][1:] #get the filename extenstion
        if ext == "png" or ext == "jpg" or ext == "bmp" or ext == "tiff" or ext == "pbm":
            print infile

            img=cv2.imread(infile,1)
            if img == None:
                continue
            sel = (0,0,0,0)
            drag_start = None
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("gray",gray)
            if (cv2.waitKey() & 255) == 27:
                break
    cv2.destroyAllWindows()
