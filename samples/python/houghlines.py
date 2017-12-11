#!/usr/bin/python

'''
This example illustrates how to use Hough Transform to find lines

Usage:
    houghlines.py [<image_name>]
    image argument defaults to ../data/pic1.png
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np
import sys
import math

if __name__ == '__main__':
    print(__doc__)

    try:
        fn = sys.argv[1]
    except IndexError:
        fn = "../data/pic1.png"

    src = cv.imread(fn)
    dst = cv.Canny(src, 50, 200)
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    if True: # HoughLinesP
        lines = cv.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
        a,b,c = lines.shape
        for i in range(a):
            cv.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv.LINE_AA)

    else:    # HoughLines
        lines = cv.HoughLines(dst, 1, math.pi/180.0, 50, np.array([]), 0, 0)
        if lines is not None:
            a,b,c = lines.shape
            for i in range(a):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0, y0 = a*rho, b*rho
                pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
                pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("detected lines", cdst)

    cv.imshow("source", src)
    cv.waitKey(0)
