#!/usr/bin/env python
import cv2 as cv

from tests_common import NewOpenCVTests

class stitching_test(NewOpenCVTests):

    def test_simple(self):

        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        stitcher = cv.createStitcher(False)
        (_result, pano) = stitcher.stitch((img1, img2))

        #cv.imshow("pano", pano)
        #cv.waitKey()

        self.assertAlmostEqual(pano.shape[0], 685, delta=100, msg="rows: %r" % list(pano.shape))
        self.assertAlmostEqual(pano.shape[1], 1025, delta=100, msg="cols: %r" % list(pano.shape))


class stitching_compose_panorama_test_no_args(NewOpenCVTests):

    def test_simple(self):

        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        stitcher = cv.createStitcher(False)

        stitcher.estimateTransform((img1, img2))

        result, _ = stitcher.composePanorama()

        assert result == 0


class stitching_compose_panorama_args(NewOpenCVTests):

    def test_simple(self):

        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        stitcher = cv.createStitcher(False)

        stitcher.estimateTransform((img1, img2))
        result, _ = stitcher.composePanorama((img1, img2))

        assert result == 0

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
