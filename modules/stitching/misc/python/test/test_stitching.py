#!/usr/bin/env python
import cv2 as cv
import numpy as np

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

        # Added Test for PR #21180
        self.assertIsNotNone(imgFea.keypoints)

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


class stitching_matches_info_test(NewOpenCVTests):

    def test_simple(self):
        finder = cv.ORB.create()
        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        img_feat1 = cv.detail.computeImageFeatures2(finder, img1)
        img_feat2 = cv.detail.computeImageFeatures2(finder, img2)

        matcher = cv.detail.BestOf2NearestMatcher_create()
        matches_info = matcher.apply(img_feat1, img_feat2)

        self.assertIsNotNone(matches_info.matches)
        self.assertIsNotNone(matches_info.inliers_mask)

class stitching_range_matcher_test(NewOpenCVTests):

    def test_simple(self):
        images = [
            self.get_sample('stitching/a1.png'),
            self.get_sample('stitching/a2.png'),
            self.get_sample('stitching/a3.png')
        ]

        orb = cv.ORB_create()

        features = [cv.detail.computeImageFeatures2(orb, img) for img in images]

        matcher = cv.detail_BestOf2NearestRangeMatcher(range_width=1)
        matches = matcher.apply2(features)

        # matches[1] is image 0 and image 1, should have non-zero confidence
        self.assertNotEqual(matches[1].confidence, 0)

        # matches[2] is image 0 and image 2, should have zero confidence due to range_width=1
        self.assertEqual(matches[2].confidence, 0)


class stitching_seam_finder_graph_cuts(NewOpenCVTests):

    def test_simple(self):
        images = [
            self.get_sample('stitching/a1.png'),
            self.get_sample('stitching/a2.png'),
            self.get_sample('stitching/a3.png')
        ]

        images = [cv.resize(img, [100, 100]) for img in images]

        finder = cv.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
        masks = [cv.UMat(255 * np.ones((img.shape[0], img.shape[1]), np.uint8)) for img in images]
        images_f = [img.astype(np.float32) for img in images]
        masks_warped = finder.find(images_f, [(0, 0), (75, 0), (150, 0)], masks)

        self.assertIsNotNone(masks_warped)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
