#!/usr/bin/env python

'''
plots image as logPolar and linearPolar

Usage:
    logpolar.py

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv

if __name__ == '__main__':
    print(__doc__)

    import sys
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'fruits.jpg'

    img = cv.imread(cv.samples.findFile(fn))
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    img2 = cv.logPolar(img, (img.shape[0]/2, img.shape[1]/2), 40, cv.WARP_FILL_OUTLIERS)
    img3 = cv.linearPolar(img, (img.shape[0]/2, img.shape[1]/2), 40, cv.WARP_FILL_OUTLIERS)

    cv.imshow('before', img)
    cv.imshow('logpolar', img2)
    cv.imshow('linearpolar', img3)

    cv.waitKey(0)
