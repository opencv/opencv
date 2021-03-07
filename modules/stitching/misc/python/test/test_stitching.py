#!/usr/bin/env python
import cv2 as cv

from tests_common import NewOpenCVTests

class stitching_test(NewOpenCVTests):

    def test_simple(self):

        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
        (_result, pano) = stitcher.stitch((img1, img2))

        #cv.imshow("pano", pano)
        #cv.waitKey()

        self.assertAlmostEqual(pano.shape[0], 685, delta=100, msg="rows: %r" % list(pano.shape))
        self.assertAlmostEqual(pano.shape[1], 1025, delta=100, msg="cols: %r" % list(pano.shape))


class stitching_detail_test(NewOpenCVTests):

    def test_simple(self):
        img = self.get_sample('stitching/a1.png')
        finder= cv.ORB.create()
        imgFea = cv.detail.computeImageFeatures2(finder,img)
        self.assertIsNotNone(imgFea)

        matcher = cv.detail_BestOf2NearestMatcher(False, 0.3)
        self.assertIsNotNone(matcher)
        matcher = cv.detail_AffineBestOf2NearestMatcher(False, False, 0.3)
        self.assertIsNotNone(matcher)
        matcher = cv.detail_BestOf2NearestRangeMatcher(2, False, 0.3)
        self.assertIsNotNone(matcher)
        estimator = cv.detail_AffineBasedEstimator()
        self.assertIsNotNone(estimator)
        estimator = cv.detail_HomographyBasedEstimator()
        self.assertIsNotNone(estimator)

        adjuster = cv.detail_BundleAdjusterReproj()
        self.assertIsNotNone(adjuster)
        adjuster = cv.detail_BundleAdjusterRay()
        self.assertIsNotNone(adjuster)
        adjuster = cv.detail_BundleAdjusterAffinePartial()
        self.assertIsNotNone(adjuster)
        adjuster = cv.detail_NoBundleAdjuster()
        self.assertIsNotNone(adjuster)

        compensator=cv.detail.ExposureCompensator_createDefault(cv.detail.ExposureCompensator_NO)
        self.assertIsNotNone(compensator)
        compensator=cv.detail.ExposureCompensator_createDefault(cv.detail.ExposureCompensator_GAIN)
        self.assertIsNotNone(compensator)
        compensator=cv.detail.ExposureCompensator_createDefault(cv.detail.ExposureCompensator_GAIN_BLOCKS)
        self.assertIsNotNone(compensator)

        seam_finder = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)
        self.assertIsNotNone(seam_finder)
        seam_finder = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)
        self.assertIsNotNone(seam_finder)
        seam_finder = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)
        self.assertIsNotNone(seam_finder)

        seam_finder = cv.detail_GraphCutSeamFinder("COST_COLOR")
        self.assertIsNotNone(seam_finder)
        seam_finder = cv.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        self.assertIsNotNone(seam_finder)
        seam_finder = cv.detail_DpSeamFinder("COLOR")
        self.assertIsNotNone(seam_finder)
        seam_finder = cv.detail_DpSeamFinder("COLOR_GRAD")
        self.assertIsNotNone(seam_finder)

        blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
        self.assertIsNotNone(blender)
        blender = cv.detail.Blender_createDefault(cv.detail.Blender_FEATHER)
        self.assertIsNotNone(blender)
        blender = cv.detail.Blender_createDefault(cv.detail.Blender_MULTI_BAND)
        self.assertIsNotNone(blender)

        timelapser = cv.detail.Timelapser_createDefault(cv.detail.Timelapser_AS_IS);
        self.assertIsNotNone(timelapser)
        timelapser = cv.detail.Timelapser_createDefault(cv.detail.Timelapser_CROP);
        self.assertIsNotNone(timelapser)


class stitching_compose_panorama_test_no_args(NewOpenCVTests):

    def test_simple(self):

        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)

        stitcher.estimateTransform((img1, img2))

        result, _ = stitcher.composePanorama()

        assert result == 0


class stitching_compose_panorama_args(NewOpenCVTests):

    def test_simple(self):

        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)

        stitcher.estimateTransform((img1, img2))
        result, _ = stitcher.composePanorama((img1, img2))

        assert result == 0


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
