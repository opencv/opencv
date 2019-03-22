#!/usr/bin/env python

'''
Floodfill sample.

Usage:
  floodfill.py [<image>]

  Click on the image to set seed point

Keys:
  f     - toggle floating range
  c     - toggle 4/8 connectivity
  ESC   - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import sys

class App():

    def update(self, dummy=None):
        if self.seed_pt is None:
            cv.imshow('floodfill', self.img)
            return
        flooded = self.img.copy()
        self.mask[:] = 0
        lo = cv.getTrackbarPos('lo', 'floodfill')
        hi = cv.getTrackbarPos('hi', 'floodfill')
        flags = self.connectivity
        if self.fixed_range:
            flags |= cv.FLOODFILL_FIXED_RANGE
        cv.floodFill(flooded, self.mask, self.seed_pt, (255, 255, 255), (lo,)*3, (hi,)*3, flags)
        cv.circle(flooded, self.seed_pt, 2, (0, 0, 255), -1)
        cv.imshow('floodfill', flooded)

    def onmouse(self, event, x, y, flags, param):
        if flags & cv.EVENT_FLAG_LBUTTON:
            self.seed_pt = x, y
            self.update()

    def run(self):
        try:
            fn = sys.argv[1]
        except:
            fn = 'fruits.jpg'

        self.img = cv.imread(cv.samples.findFile(fn))
        if self.img is None:
            print('Failed to load image file:', fn)
            sys.exit(1)

        h, w = self.img.shape[:2]
        self.mask = np.zeros((h+2, w+2), np.uint8)
        self.seed_pt = None
        self.fixed_range = True
        self.connectivity = 4

        self.update()
        cv.setMouseCallback('floodfill', self.onmouse)
        cv.createTrackbar('lo', 'floodfill', 20, 255, self.update)
        cv.createTrackbar('hi', 'floodfill', 20, 255, self.update)

        while True:
            ch = cv.waitKey()
            if ch == 27:
                break
            if ch == ord('f'):
                self.fixed_range = not self.fixed_range
                print('using %s range' % ('floating', 'fixed')[self.fixed_range])
                self.update()
            if ch == ord('c'):
                self.connectivity = 12-self.connectivity
                print('connectivity =', self.connectivity)
                self.update()

        print('Done')


if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()
