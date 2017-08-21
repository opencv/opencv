#!/usr/bin/env python
import cv2

from tests_common import NewOpenCVTests

class stitching_test(NewOpenCVTests):

    def test_simple(self):

        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        stitcher = cv2.createStitcher(False)
        (result, pano) = stitcher.stitch((img1, img2))

        #cv2.imshow("pano", pano)
        #cv2.waitKey()

        self.assertAlmostEqual(pano.shape[0], 685, delta=100, msg="rows: %r" % list(pano.shape))
        self.assertAlmostEqual(pano.shape[1], 1025, delta=100, msg="cols: %r" % list(pano.shape))
