#!/usr/bin/python

'''
This example illustrates how to use cv.HoughCircles() function.
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np
import sys
from numpy import pi, sin, cos

from tests_common import NewOpenCVTests

def circleApproximation(circle):

    nPoints = 30
    dPhi = 2*pi / nPoints
    contour = []
    for i in range(nPoints):
        contour.append(([circle[0] + circle[2]*cos(i*dPhi),
            circle[1] + circle[2]*sin(i*dPhi)]))

    return np.array(contour).astype(int)

def convContoursIntersectiponRate(c1, c2):

    s1 = cv.contourArea(c1)
    s2 = cv.contourArea(c2)

    s, _ = cv.intersectConvexConvex(c1, c2)

    return 2*s/(s1+s2)

class houghcircles_test(NewOpenCVTests):

    def test_houghcircles(self):

        fn = "samples/data/board.jpg"

        src = self.get_sample(fn, 1)
        img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        img = cv.medianBlur(img, 5)

        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)[0]

        testCircles = [[38, 181, 17.6],
        [99.7, 166, 13.12],
        [142.7, 160, 13.52],
        [223.6, 110, 8.62],
        [79.1, 206.7, 8.62],
        [47.5, 351.6, 11.64],
        [189.5, 354.4, 11.64],
        [189.8, 298.9, 10.64],
        [189.5, 252.4, 14.62],
        [252.5, 393.4, 15.62],
        [602.9, 467.5, 11.42],
        [222, 210.4, 9.12],
        [263.1, 216.7, 9.12],
        [359.8, 222.6, 9.12],
        [518.9, 120.9, 9.12],
        [413.8, 113.4, 9.12],
        [489, 127.2, 9.12],
        [448.4, 121.3, 9.12],
        [384.6, 128.9, 8.62]]

        matches_counter = 0

        for i in range(len(testCircles)):
            for j in range(len(circles)):

                tstCircle = circleApproximation(testCircles[i])
                circle = circleApproximation(circles[j])
                if convContoursIntersectiponRate(tstCircle, circle) > 0.6:
                    matches_counter += 1

        self.assertGreater(float(matches_counter) / len(testCircles), .5)
        self.assertLess(float(len(circles) - matches_counter) / len(circles), .75)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
