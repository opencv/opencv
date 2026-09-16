#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests


class linear_assignment_test(NewOpenCVTests):

    def test_basic(self):
        cost = np.array([[4, 1, 3],
                         [2, 0, 5],
                         [3, 2, 2]], dtype=np.float64)
        total, assignment = cv.linearAssignment(cost)

        self.assertAlmostEqual(total, 5.0, places=9)
        self.assertEqual(list(assignment), [1, 0, 2])

    def test_non_square(self):
        cost = np.array([[7, 1, 9, 8],
                         [6, 5, 2, 4]], dtype=np.float32)
        total, assignment = cv.linearAssignment(cost)

        self.assertEqual(len(assignment), 2)
        self.assertAlmostEqual(total, 3.0, places=5)

    def test_threshold_forbids_expensive_pairs(self):
        # Both pairs are affordable but cost 20 together, while leaving one row unmatched
        # costs 10. The cheap pair alone wins.
        cost = np.array([[0, 10],
                         [10, 100]], dtype=np.float64)

        total, assignment = cv.linearAssignment(cost, costThreshold=10.0)
        self.assertEqual(list(assignment), [0, -1])
        self.assertAlmostEqual(total, 0.0, places=9)

        total, assignment = cv.linearAssignment(cost, costThreshold=50.0)
        self.assertEqual(sum(1 for a in assignment if a >= 0), 2)
        self.assertAlmostEqual(total, 20.0, places=9)

    def test_infinity_forbids_a_pair(self):
        cost = np.array([[10, np.inf],
                         [1, np.inf]], dtype=np.float64)
        total, assignment = cv.linearAssignment(cost)

        self.assertEqual(list(assignment), [-1, 0])
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_empty(self):
        total, assignment = cv.linearAssignment(np.zeros((0, 0), dtype=np.float64))
        self.assertAlmostEqual(total, 0.0, places=9)
        self.assertEqual(len(assignment), 0)

    def test_matches_scipy_style_reference(self):
        # Without a threshold this is the plain optimal assignment, so a brute force over
        # permutations is an exact reference.
        from itertools import permutations

        rng = np.random.default_rng(12345)
        for _ in range(50):
            n = int(rng.integers(1, 6))
            cost = rng.uniform(-5.0, 20.0, size=(n, n))
            expected = min(sum(cost[i, p[i]] for i in range(n))
                           for p in permutations(range(n)))

            total, assignment = cv.linearAssignment(cost)
            self.assertAlmostEqual(total, expected, places=9)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
