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

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = '../data/fruits.jpg'
    print(__doc__)

    img = cv.imread(fn, True)
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    seed_pt = None
    fixed_range = True
    connectivity = 4

    def update(dummy=None):
        if seed_pt is None:
            cv.imshow('floodfill', img)
            return
        flooded = img.copy()
        mask[:] = 0
        lo = cv.getTrackbarPos('lo', 'floodfill')
        hi = cv.getTrackbarPos('hi', 'floodfill')
        flags = connectivity
        if fixed_range:
            flags |= cv.FLOODFILL_FIXED_RANGE
        cv.floodFill(flooded, mask, seed_pt, (255, 255, 255), (lo,)*3, (hi,)*3, flags)
        cv.circle(flooded, seed_pt, 2, (0, 0, 255), -1)
        cv.imshow('floodfill', flooded)

    def onmouse(event, x, y, flags, param):
        global seed_pt
        if flags & cv.EVENT_FLAG_LBUTTON:
            seed_pt = x, y
            update()

    update()
    cv.setMouseCallback('floodfill', onmouse)
    cv.createTrackbar('lo', 'floodfill', 20, 255, update)
    cv.createTrackbar('hi', 'floodfill', 20, 255, update)

    while True:
        ch = cv.waitKey()
        if ch == 27:
            break
        if ch == ord('f'):
            fixed_range = not fixed_range
            print('using %s range' % ('floating', 'fixed')[fixed_range])
            update()
        if ch == ord('c'):
            connectivity = 12-connectivity
            print('connectivity =', connectivity)
            update()
    cv.destroyAllWindows()
