#!/usr/bin/env python

'''
browse.py
=========

Sample shows how to implement a simple hi resolution image navigation

Usage
-----
browse.py [image filename]

'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

# built-in modules
import sys

def main():
    if len(sys.argv) > 1:
        fn = cv.samples.findFile(sys.argv[1])
        print('loading %s ...' % fn)
        img = cv.imread(fn)
        if img is None:
            print('Failed to load fn:', fn)
            sys.exit(1)

    else:
        sz = 4096
        print('generating %dx%d procedural image ...' % (sz, sz))
        img = np.zeros((sz, sz), np.uint8)
        track = np.cumsum(np.random.rand(500000, 2)-0.5, axis=0)
        track = np.int32(track*10 + (sz/2, sz/2))
        cv.polylines(img, [track], 0, 255, 1, cv.LINE_AA)


    small = img
    for _i in xrange(3):
        small = cv.pyrDown(small)

    def onmouse(event, x, y, flags, param):
        h, _w = img.shape[:2]
        h1, _w1 = small.shape[:2]
        x, y = 1.0*x*h/h1, 1.0*y*h/h1
        zoom = cv.getRectSubPix(img, (800, 600), (x+0.5, y+0.5))
        cv.imshow('zoom', zoom)

    cv.imshow('preview', small)
    cv.setMouseCallback('preview', onmouse)
    cv.waitKey()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
