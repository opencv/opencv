#!/usr/bin/python

'''
This example illustrates how to use Hough Transform to find lines

Usage:
    houghlines.py [<image_name>]
    image argument defaults to pic1.png
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np

import sys
import math

def main():
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'pic1.png'

    src = cv.imread(cv.samples.findFile(fn))
    dst = cv.Canny(src, 50, 200)
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    if True: # HoughLinesP
        lines = cv.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
        if lines is not None:
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                cv.line(cdst, (x1, y1), (x2, y2), (0, 0, 255), 3, cv.LINE_AA)

    else:    # HoughLines
        lines = cv.HoughLines(dst, 1, math.pi/180.0, 50, np.array([]), 0, 0)
        if lines is not None:
            for rho, theta in np.asarray(lines).reshape(-1, 2):
                a = math.cos(theta)
                b = math.sin(theta)
                x0, y0 = a*rho, b*rho
                pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
                pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
                
    cv.imshow("detected lines", cdst)

    cv.imshow("source", src)
    cv.waitKey(0)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
