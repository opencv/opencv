#!/usr/bin/env python

'''
Morphology operations.

Usage:
  morphology.py [<image>]

Keys:
  1   - change operation
  2   - change structure element shape
  ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

import numpy as np
import cv2 as cv


def main():
    import sys
    from itertools import cycle
    from common import draw_str

    try:
        fn = sys.argv[1]
    except:
        fn = 'baboon.jpg'

    img = cv.imread(cv.samples.findFile(fn))

    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    cv.imshow('original', img)

    modes = cycle(['erode/dilate', 'open/close', 'blackhat/tophat', 'gradient'])
    str_modes = cycle(['ellipse', 'rect', 'cross'])

    if PY3:
        cur_mode = next(modes)
        cur_str_mode = next(str_modes)
    else:
        cur_mode = modes.next()
        cur_str_mode = str_modes.next()

    def update(dummy=None):
        sz = cv.getTrackbarPos('op/size', 'morphology')
        iters = cv.getTrackbarPos('iters', 'morphology')
        opers = cur_mode.split('/')
        if len(opers) > 1:
            sz = sz - 10
            op = opers[sz > 0]
            sz = abs(sz)
        else:
            op = opers[0]
        sz = sz*2+1

        str_name = 'MORPH_' + cur_str_mode.upper()
        oper_name = 'MORPH_' + op.upper()
        st = cv.getStructuringElement(getattr(cv, str_name), (sz, sz))
        res = cv.morphologyEx(img, getattr(cv, oper_name), st, iterations=iters)

        draw_str(res, (10, 20), 'mode: ' + cur_mode)
        draw_str(res, (10, 40), 'operation: ' + oper_name)
        draw_str(res, (10, 60), 'structure: ' + str_name)
        draw_str(res, (10, 80), 'ksize: %d  iters: %d' % (sz, iters))
        cv.imshow('morphology', res)

    cv.namedWindow('morphology')
    cv.createTrackbar('op/size', 'morphology', 12, 20, update)
    cv.createTrackbar('iters', 'morphology', 1, 10, update)
    update()
    while True:
        ch = cv.waitKey()
        if ch == 27:
            break
        if ch == ord('1'):
            if PY3:
                cur_mode = next(modes)
            else:
                cur_mode = modes.next()
        if ch == ord('2'):
            if PY3:
                cur_str_mode = next(str_modes)
            else:
                cur_str_mode = str_modes.next()
        update()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
