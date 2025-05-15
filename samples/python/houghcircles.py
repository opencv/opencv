#!/usr/bin/python

'''
This example illustrates how to use cv.HoughCircles() function.

Usage:
    houghcircles.py [<image_name>]
    image argument defaults to board.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import sys

def main():
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'board.jpg'

    src = cv.imread(cv.samples.findFile(fn))
    img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img, 5)
    cimg = src.copy() # numpy function

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 200, 30, 5, 30)

    if circles is not None: # Check if circles have been found and only then iterate over these and add them to the image
        circles = np.uint16(np.around(circles))
        _a, b, _c = circles.shape
        for i in range(b):
            cv.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv.LINE_AA)
            cv.circle(cimg, (circles[0][i][0], circles[0][i][1]), 2, (0, 255, 0), 3, cv.LINE_AA)  # draw center of circle

        cv.imshow("detected circles", cimg)

    cv.imshow("source", src)
    cv.waitKey(0)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
