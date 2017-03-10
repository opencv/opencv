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
            None, #Default argument
            0, (0), (0, 0), (0, 0, 0), (0, 0, 0, 0),
            np.array([0]), np.array([0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0, 0]),
            np.zeros((1, 1)), np.zeros((1, 2)), np.zeros((1, 3)), np.zeros((1, 4)),
            np.zeros((2, 1)), np.zeros((3, 1)), np.zeros((4, 1)),
        ]

        self.invalid_scalars = [
            '', (0, ''), (0, 0, 0, 0, 0),
        ]

        self.scalar_values = range(0, 256)

        self.width = 10
        self.pt1 = (0, 0)
        self.pt2 = (self.width - 1, self.width - 1)
        self.img = np.ones((self.width, self.width), np.uint8)
        self.returned_scalar = cv2.mean(self.img)

    def test_types_setting_valid_scalars(self):

        for scalar in self.valid_scalars:
            try:
                cv2.line(self.img, self.pt1, self.pt2, scalar)
            except:
                self.assertTrue(False,
                    "Valid scalar was rejected!"
                    "\tType: {0}\n"
                    "\tValue: {1}".format(type(scalar).__name__, scalar))
            self.assertTrue(np.trace(self.img) == 0, "Unexpected trace value")
            expected_count = self.width * (self.width - 1)
            count = np.count_nonzero(self.img)
            self.assertTrue(count == expected_count,
                "Expected non-zero elements count of {0}, got {1}".format(expected_count, count))

    def test_types_setting_invalid_scalars(self):

        for scalar in self.invalid_scalars:
            try:
                cv2.line(self.img, self.pt1, self.pt2, scalar)
            except:
                continue
            self.assertTrue(False,
                "Invalid scalar was accepted!\n"
                "\tType: {0}\n"
                "\tValue: {1}".format(type(scalar).__name__, scalar))

    def test_types_getter(self):

        length = len(self.returned_scalar)
        self.assertTrue(length == 4,
            "Expected scalar length of 4, got {0}".format(length))
        for i in range(4):
            expected_mean = (i == 0) * 1
            self.assertTrue(self.returned_scalar[i] == expected_mean,
                "Expected mean value of {0}, got {1}".format(expected_mean, self.returned_scalar[i]))

    def test_types_setter(self):
        try:
            self.test_types_setting_valid_scalars()
        except:
            self.skipTest("Expected setting valid scalars to pass first")

        for scalar in self.scalar_values:
            cv2.line(self.img, self.pt1, self.pt2, scalar)
            expected_trace = self.width * scalar
            expected_count = self.width * (self.width if scalar else self.width - 1)
            trace = np.trace(self.img)
            count = np.count_nonzero(self.img)
            self.assertTrue(trace == expected_trace,
                "Expected trace value of {0}, got {1}".format(expected_trace, trace))
            self.assertTrue(count == expected_count,
                "Expected non-zero elements count of {0}, got {1}".format(expected_count, count))

if __name__ == '__main__':
    import unittest
    unittest.main()
