#!/usr/bin/env python

'''
Utility for measuring python opencv API coverage by samples.
'''

# Python 2/3 compatibility
from __future__ import print_function

from glob import glob
import cv2 as cv
import re

if __name__ == '__main__':
    cv2_callable = set(['cv.'+name for name in dir(cv) if callable( getattr(cv, name) )])

    found = set()
    for fn in glob('*.py'):
        print(' --- ', fn)
        code = open(fn).read()
        found |= set(re.findall('cv2?\.\w+', code))

    cv2_used = found & cv2_callable
    cv2_unused = cv2_callable - cv2_used
    with open('unused_api.txt', 'w') as f:
        f.write('\n'.join(sorted(cv2_unused)))

    r = 1.0 * len(cv2_used) / len(cv2_callable)
    print('\ncv api coverage: %d / %d  (%.1f%%)' % ( len(cv2_used), len(cv2_callable), r*100 ))
