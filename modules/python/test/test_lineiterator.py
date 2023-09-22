#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv

from tests_common import NewOpenCVTests


class lineiterator_test(NewOpenCVTests):

    def test_lineiterator(self):
        n = 10
        orig = 0
        p1 = (orig, orig)
        p2 = (orig + n, orig + n)
        d1 = cv.LineIterator(p1, p2)
        self.assertEqual(d1.next(), True)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
