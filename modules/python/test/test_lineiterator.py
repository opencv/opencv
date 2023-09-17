#!/usr/bin/env python


# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np
import sys

from tests_common import NewOpenCVTests


class lineiterator_test(NewOpenCVTests):

    def test_lineiterator(self):

        p1 = (0, 0)
        p2 = (100, 100)
        d1 = cv.LineIterator(p1, p2)
        self.assertEqual(d1.pos(), p1)
        self.assertEqual(d1.next(), True)
        self.assertEqual(d1.pos(), (1.0, 1.0))
        d2 = cv.LineIterator(p2, p1)
        self.assertEqual(d2.pos(), p2)
        self.assertEqual(d2.next(), True)
        self.assertEqual(d2.pos(), (99.0, 99.0))
        count = 0
        for p in cv.LineIterator(p1, p2):
            count = count + 1
            if count == 10:
                self.assertEqual(p.pos(), (10.0, 10.0))
                break

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
