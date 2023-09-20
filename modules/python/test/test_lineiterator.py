#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import sys
import cv2 as cv

from tests_common import NewOpenCVTests

try:
    if sys.version_info[:2] < (3, 0):
        raise unittest.SkipTest('Python 2.x is not supported')

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

except unittest.SkipTest as e:

    message = str(e)

    class TestSkip(unittest.TestCase):
        def setUp(self):
            self.skipTest('Skip tests: ' + message)

        def test_skip():
            pass

    pass



if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
