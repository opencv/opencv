#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
import sys
import cv2 as cv

from tests_common import NewOpenCVTests


class lineiterator_test(NewOpenCVTests):

    def test_lineiterator(self):
        n = 5
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
            line_p21 = iter(cv.LineIterator(p2, p1))
            for point21 in line_p21:
                self.assertEqual(point21, (orig + line_p21.count - count - 1,
                                           orig + line_p21.count - count - 1))
                count = count + 1
            self.assertEqual(line_p21.count, line_p21.iter)
            self.assertEqual(line_p21.count, count)
            line_p21 = iter(cv.LineIterator(p2, p1))
            count = 0
            for point21 in line_p21:
                self.assertEqual(point21, (orig + line_p21.count - count - 1,
                                           orig + line_p21.count - count - 1))
                count = count + 1


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
