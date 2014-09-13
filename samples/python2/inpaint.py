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

import numpy as np
import cv2
from common import Sketcher

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = '../data/fruits.jpg'

    print __doc__

    img = cv2.imread(fn)
    if img is None:
        print 'Failed to load image file:', fn
        sys.exit(1)

    img_mark = img.copy()
    mark = np.zeros(img.shape[:2], np.uint8)
    sketch = Sketcher('img', [img_mark, mark], lambda : ((255, 255, 255), 255))

    while True:
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            res = cv2.inpaint(img_mark, mark, 3, cv2.INPAINT_TELEA)
            cv2.imshow('inpaint', res)
        if ch == ord('r'):
            img_mark[:] = img
            mark[:] = 0
            sketch.show()
    cv2.destroyAllWindows()
