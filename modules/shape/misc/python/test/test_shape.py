#!/usr/bin/env python
import cv2 as cv

from tests_common import NewOpenCVTests

class shape_test(NewOpenCVTests):

    def test_computeDistance(self):

        a = self.get_sample('samples/data/shape_sample/1.png', cv.IMREAD_GRAYSCALE)
        b = self.get_sample('samples/data/shape_sample/2.png', cv.IMREAD_GRAYSCALE)

        _, ca, _ = cv.findContours(a, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
        _, cb, _ = cv.findContours(b, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)

        hd = cv.createHausdorffDistanceExtractor()
        sd = cv.createShapeContextDistanceExtractor()

        d1 = hd.computeDistance(ca[0], cb[0])
        d2 = sd.computeDistance(ca[0], cb[0])

        self.assertAlmostEqual(d1, 26.4196891785, 3, "HausdorffDistanceExtractor")
        self.assertAlmostEqual(d2, 0.25804194808, 3, "ShapeContextDistanceExtractor")

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
