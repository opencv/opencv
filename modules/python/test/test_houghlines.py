#!/usr/bin/python

'''
This example illustrates how to use Hough Transform to find lines
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np
import sys
import math

from tests_common import NewOpenCVTests

def linesDiff(line1, line2):

    norm1 = cv2.norm(line1 - line2, cv2.NORM_L2)
    line3 = line1[2:4] + line1[0:2]
    norm2 = cv2.norm(line3 - line2, cv2.NORM_L2)

    return min(norm1, norm2)

class houghlines_test(NewOpenCVTests):

    def test_houghlines(self):

        fn = "/samples/data/pic1.png"

        src = self.get_sample(fn)
        dst = cv2.Canny(src, 50, 200)

        lines = cv2.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)[:,0,:]

        eps = 5
        testLines = [
            #rect1
             [ 232,  25, 43, 25],
             [ 43, 129, 232, 129],
             [ 43, 129,  43,  25],
             [232, 129, 232,  25],
            #rect2
             [251,  86, 314, 183],
             [252,  86, 323,  40],
             [315, 183, 386, 137],
             [324,  40, 386, 136],
            #triangle
             [245, 205, 377, 205],
             [244, 206, 305, 278],
             [306, 279, 377, 205],
            #rect3
             [153, 177, 196, 177],
             [153, 277, 153, 179],
             [153, 277, 196, 277],
             [196, 177, 196, 277]]

        matches_counter = 0

        for i in range(len(testLines)):
            for j in range(len(lines)):
                if linesDiff(testLines[i], lines[j]) < eps:
                    matches_counter += 1

        self.assertGreater(float(matches_counter) / len(testLines), .7)