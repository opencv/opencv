#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
import sys
import cv2 as cv

from tests_common import NewOpenCVTests


class lineiterator_test(NewOpenCVTests):

    def test_lineiterator(self):
        n = 10
        orig = 0
        p1 = (orig, orig)
        p2 = (orig + n, orig + n)
        d1 = cv.LineIterator(p1, p2)
        self.assertEqual(d1.count, n + 1)
        if sys.version_info[0] >= 3:
            count = 0
            for p in d1:
                self.assertEqual(p, (count, count))
                count = count + 1
            self.assertEqual(d1.count, d1.iter)
            count = 0
            d1 = iter(cv.LineIterator(p1, p2))
            d2 = iter(cv.LineIterator(p2, p1))
            for p1, p2 in zip(d1, d2):
                self.assertEqual(p1, (orig + count, orig + count))
                self.assertEqual(p2, (orig + d2.count - count - 1,
                                      orig + d2.count - count - 1))
                count = count + 1
            self.assertEqual(d1.count, d1.iter)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
