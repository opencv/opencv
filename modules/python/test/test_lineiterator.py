#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv

from tests_common import NewOpenCVTests


class lineiterator_test(NewOpenCVTests):

    def test_lineiterator(self):
        n = 10
        p1 = (0, 0)
        p2 = (n, n)
        d1 = cv.LineIterator(p1, p2)
        self.assertEqual(d1.count, n + 1)
        count = 0
        for p in d1:
            self.assertEqual(p, (count, count))
            count = count + 1
        self.assertEqual(d1.count, d1.iter)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
