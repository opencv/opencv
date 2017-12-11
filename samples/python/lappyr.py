#!/usr/bin/env python

''' An example of Laplacian Pyramid construction and merging.

Level : Intermediate

Usage : python lappyr.py [<video source>]

References:
  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.54.299

Alexander Mordvintsev 6/10/12
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv
import video
from common import nothing, getsize

def build_lappyr(img, leveln=6, dtype=np.int16):
    img = dtype(img)
    levels = []
    for _i in xrange(leveln-1):
        next_img = cv.pyrDown(img)
        img1 = cv.pyrUp(next_img, dstsize=getsize(img))
        levels.append(img-img1)
        img = next_img
    levels.append(img)
    return levels

def merge_lappyr(levels):
    img = levels[-1]
    for lev_img in levels[-2::-1]:
        img = cv.pyrUp(img, dstsize=getsize(lev_img))
        img += lev_img
    return np.uint8(np.clip(img, 0, 255))


if __name__ == '__main__':
    import sys
    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0
    cap = video.create_capture(fn)

    leveln = 6
    cv.namedWindow('level control')
    for i in xrange(leveln):
        cv.createTrackbar('%d'%i, 'level control', 5, 50, nothing)

    while True:
        ret, frame = cap.read()

        pyr = build_lappyr(frame, leveln)
        for i in xrange(leveln):
            v = int(cv.getTrackbarPos('%d'%i, 'level control') / 5)
            pyr[i] *= v
        res = merge_lappyr(pyr)

        cv.imshow('laplacian pyramid filter', res)

        if cv.waitKey(1) == 27:
            break
