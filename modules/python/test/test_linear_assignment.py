#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests


class linear_assignment_test(NewOpenCVTests):

    def check_pairs(self, cost, total, assignment, threshold=None):
        '''Every reported pair must be legal, each column used once, and the total must match.'''
        self.assertEqual(len(assignment), cost.shape[0])
        seen = set()
        summed = 0.0
        for row, col in enumerate(assignment):
            if col < 0:
                self.assertEqual(col, -1)
                continue
            self.assertLess(col, cost.shape[1])
            self.assertNotIn(col, seen)
            seen.add(col)
            self.assertTrue(np.isfinite(cost[row, col]))
            if threshold is not None:
                self.assertLessEqual(cost[row, col], threshold)
            summed += cost[row, col]
        self.assertAlmostEqual(total, summed, places=9)

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

        # Row 0 is cheapest on column 1 and row 1 on column 2, so both rows are matched.
        self.assertEqual(list(assignment), [1, 2])
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
        self.assertEqual(list(assignment), [1, 0])
        self.assertAlmostEqual(total, 20.0, places=9)

    def test_infinity_forbids_a_pair(self):
        cost = np.array([[10, np.inf],
                         [1, np.inf]], dtype=np.float64)
        total, assignment = cv.linearAssignment(cost)

        # Column 1 is forbidden on both rows, so only one row can be matched and the solver
        # keeps the cheaper of the two. Row 1 costs 1 against row 0's 10, so row 0 is the one
        # left at -1.
        self.assertEqual(list(assignment), [-1, 0])
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_empty_input_throws(self):
        with self.assertRaises(cv.error):
            cv.linearAssignment(np.zeros((0, 0), dtype=np.float64))
        with self.assertRaises(cv.error):
            cv.linearAssignment(np.zeros((3, 0), dtype=np.float64))

    def test_matches_bruteforce_reference(self):
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
            self.check_pairs(cost, total, assignment)
            self.assertTrue(all(a >= 0 for a in assignment))


    def test_matches_lapjv(self):
        # Optional cross-check against a reference implementation. lapjv takes square matrices
        # only and has no threshold, which is exactly the unconstrained case. Skipped when the
        # package is not installed.
        try:
            from lapjv import lapjv
        except ImportError:
            self.skipTest('lapjv is not installed')

        rng = np.random.default_rng(987)
        for _ in range(25):
            n = int(rng.integers(2, 12))
            cost = rng.uniform(-5.0, 20.0, size=(n, n))

            total, assignment = cv.linearAssignment(cost)
            ref_cols = lapjv(cost)[0]
            ref_total = sum(cost[i, ref_cols[i]] for i in range(n))

            # Equal totals is the real check. The pairing itself is only unique when no two
            # matchings tie, which random costs make overwhelmingly likely but not certain.
            self.assertAlmostEqual(total, ref_total, places=9)
            self.check_pairs(cost, total, assignment)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
