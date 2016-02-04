#!/usr/bin/env python

'''
example to detect upright people in images using HOG features
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def intersectionRate(s1, s2):

    x1, y1, x2, y2 = s1
    s1 = [[x1, y1], [x2,y1], [x2, y2], [x1, y2] ]

    x1, y1, x2, y2 = s2
    s2 = [[x1, y1], [x2,y1], [x2, y2], [x1, y2] ]
    area, intersection = cv2.intersectConvexConvex(np.array(s1), np.array(s2))

    return 2 * area / (cv2.contourArea(np.array(s1)) + cv2.contourArea(np.array(s2)))

from tests_common import NewOpenCVTests

class peopledetect_test(NewOpenCVTests):
    def test_peopledetect(self):

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

        dirPath = 'samples/data/'
        samples = ['basketball1.png', 'basketball2.png']

        testPeople = [
        [[23, 76, 164, 477], [440, 22, 637, 478]],
        [[23, 76, 164, 477], [440, 22, 637, 478]]
        ]

        eps = 0.5

        for sample in samples:

            img = self.get_sample(dirPath + sample, 0)

            found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
            found_filtered = []
            for ri, r in enumerate(found):
                for qi, q in enumerate(found):
                    if ri != qi and inside(r, q):
                        break
                else:
                    found_filtered.append(r)

            matches = 0

            for i in range(len(found_filtered)):
                for j in range(len(testPeople)):

                    found_rect = (found_filtered[i][0], found_filtered[i][1],
                        found_filtered[i][0] + found_filtered[i][2],
                        found_filtered[i][1] + found_filtered[i][3])

                    if intersectionRate(found_rect, testPeople[j][0]) > eps or intersectionRate(found_rect, testPeople[j][1]) > eps:
                        matches += 1

            self.assertGreater(matches, 0)