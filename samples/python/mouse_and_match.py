#!/usr/bin/env python
'''
mouse_and_match.py [-i path | --input path: default ../data/]

Demonstrate using a mouse to interact with an image:
 Read in the images in a directory one by one
 Allow the user to select parts of an image with a mouse
 When they let go of the mouse, it correlates (using matchTemplate) that patch with the image.

 SPACE for next image
 ESC to exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# built-in modules
import os
import sys
import glob
import argparse
from math import *


class App():
    drag_start = None
    sel = (0,0,0,0)

    def onmouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = x, y
            self.sel = (0,0,0,0)
        elif event == cv.EVENT_LBUTTONUP:
            if self.sel[2] > self.sel[0] and self.sel[3] > self.sel[1]:
                patch = self.gray[self.sel[1]:self.sel[3], self.sel[0]:self.sel[2]]
                result = cv.matchTemplate(self.gray, patch, cv.TM_CCOEFF_NORMED)
                result = np.abs(result)**3
                _val, result = cv.threshold(result, 0.01, 0, cv.THRESH_TOZERO)
                result8 = cv.normalize(result, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
                cv.imshow("result", result8)
            self.drag_start = None
        elif self.drag_start:
            #print flags
            if flags & cv.EVENT_FLAG_LBUTTON:
                minpos = min(self.drag_start[0], x), min(self.drag_start[1], y)
                maxpos = max(self.drag_start[0], x), max(self.drag_start[1], y)
                self.sel = (minpos[0], minpos[1], maxpos[0], maxpos[1])
                img = cv.cvtColor(self.gray, cv.COLOR_GRAY2BGR)
                cv.rectangle(img, (self.sel[0], self.sel[1]), (self.sel[2], self.sel[3]), (0,255,255), 1)
                cv.imshow("gray", img)
            else:
                print("selection is complete")
                self.drag_start = None

    def run(self):
        parser = argparse.ArgumentParser(description='Demonstrate mouse interaction with images')
        parser.add_argument("-i","--input", default='../data/', help="Input directory.")
        args = parser.parse_args()
        path = args.input

        cv.namedWindow("gray",1)
        cv.setMouseCallback("gray", self.onmouse)
        '''Loop through all the images in the directory'''
        for infile in glob.glob( os.path.join(path, '*.*') ):
            ext = os.path.splitext(infile)[1][1:] #get the filename extension
            if ext == "png" or ext == "jpg" or ext == "bmp" or ext == "tiff" or ext == "pbm":
                print(infile)

                img = cv.imread(infile, cv.IMREAD_COLOR)
                if img is None:
                    continue
                self.sel = (0,0,0,0)
                self.drag_start = None
                self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                cv.imshow("gray", self.gray)
                if cv.waitKey() == 27:
                    break

        print('Done')


if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()
