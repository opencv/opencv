#!/usr/bin/env python

'''
Inpainting sample.

Inpainting repairs damage to images by floodfilling
the damage with surrounding image areas.

Usage:
  inpaint.py [<image>]

Keys:
  SPACE - inpaint
  r     - reset the inpainting mask
  ESC   - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from common import Sketcher

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'fruits.jpg'

    print(__doc__)

    img = cv.imread(cv.samples.findFile(fn))
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    img_mark = img.copy()
    mark = np.zeros(img.shape[:2], np.uint8)
    sketch = Sketcher('img', [img_mark, mark], lambda : ((255, 255, 255), 255))

    while True:
        ch = cv.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            res = cv.inpaint(img_mark, mark, 3, cv.INPAINT_TELEA)
            cv.imshow('inpaint', res)
        if ch == ord('r'):
            img_mark[:] = img
            mark[:] = 0
            sketch.show()
    cv.destroyAllWindows()
