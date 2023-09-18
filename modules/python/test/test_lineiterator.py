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
        p2 = (10, 10)
        d1 = cv.LineIterator(p1, p2)
        self.assertEqual(d1.count, 11)
        for p in d1:
            pass
        d2 = cv.LineIterator(p2, p1)
        count = 0
        seg = cv.LineIterator(p1, p2)
        for p in seg:
            count = count + 1
        self.assertEqual(count, seg.count)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
