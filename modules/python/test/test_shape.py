#!/usr/bin/env python
import cv2

from tests_common import NewOpenCVTests

class shape_test(NewOpenCVTests):

    def test_computeDistance(self):

        a = self.get_sample('samples/data/shape_sample/1.png', cv2.IMREAD_GRAYSCALE);
        b = self.get_sample('samples/data/shape_sample/2.png', cv2.IMREAD_GRAYSCALE);

        _, ca, _ = cv2.findContours(a, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        _, cb, _ = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

        hd = cv2.createHausdorffDistanceExtractor()
        sd = cv2.createShapeContextDistanceExtractor()

        d1 = hd.computeDistance(ca[0], cb[0])
        d2 = sd.computeDistance(ca[0], cb[0])

        self.assertAlmostEqual(d1, 26.4196891785, 3, "HausdorffDistanceExtractor")
        self.assertAlmostEqual(d2, 0.25804194808, 3, "ShapeContextDistanceExtractor")
