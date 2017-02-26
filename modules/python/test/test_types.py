#!/usr/bin/env python

'''
Vector binding types
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

from tests_common import NewOpenCVTests

class types_test(NewOpenCVTests):

    def setUp(self):
        self.valid_scalars = [
            1, (1), (1, 1), (1, 1, 1), (1, 1, 1, 1),
            np.array([1]), np.array([1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1, 1]),
            np.ones((1, 1)), np.ones((1, 2)), np.ones((1, 3)), np.ones((1, 4)),
            np.ones((2, 1)), np.ones((3, 1)), np.ones((4, 1)),
        ]

        self.invalid_scalars = [
            'str', (1, 'str'), (1, 1, 1, 1, 1),
        ]

        self.width = 10
        self.pt1 = (0, 0)
        self.pt2 = (self.width - 1, self.width - 1)
        self.img = np.zeros((self.width, self.width))
        self.returned_scalar = cv2.mean(self.img)

    def test_valid_scalars(self):

        for scalar in self.valid_scalars:
            try:
                cv2.line(self.img, self.pt1, self.pt2, scalar)
                #self.assertTrue(np.all(np.eye(self.width) == self.img))
            except:
                self.assertTrue(False), \
                    "Valid scalar is rejected!\n" \
                    "\tType: {0}\n" \
                    "\tValue: {1}".format(type(scalar).__name__, scalar)

    def test_invalid_scalars(self):

        for scalar in self.invalid_scalars:
            try:
                cv2.line(self.img, self.pt1, self.pt2, scalar)
            except:
                continue
            self.assertTrue(False), \
                "Invalid scalar is accepted!\n" \
                "\tType: {0}\n" \
                "\tValue: {1}".format(type(scalar).__name__, scalar)

    def test_returned_scalar(self):

        for i in range(4):
            try:
                self.returned_scalar[i]
            except:
                self.assertTrue(False), \
                    "Scalar index {0} does not exist!".format(i)

if __name__ == '__main__':
    import unittest
    unittest.main()
