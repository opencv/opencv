#!/usr/bin/python

'''
This example illustrates how to use cv2.HoughCircles() function.
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np
import sys

from tests_common import NewOpenCVTests

class houghcircles_test(NewOpenCVTests):

    def test_houghcircles(self):

        fn = "samples/data/board.jpg"

        src = self.get_sample(fn, 1)
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 5)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)[0]

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

        eps = 7
        matches_counter = 0

        for i in range(len(testCircles)):
            for j in range(len(circles)):
                if cv2.norm(testCircles[i] - circles[j], cv2.NORM_L2) < eps:
                    matches_counter += 1

        self.assertGreater(float(matches_counter) / len(testCircles), .5)
        self.assertLess(float(len(circles) - matches_counter) / len(circles), .7)