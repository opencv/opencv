#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

import cv2 as cv
from tests_common import NewOpenCVTests

EXPECTED_COEFFS_SIZE = 78

class ChromaticAberrationTest(NewOpenCVTests):
    def setUp(self):
        super().setUp()

        self.test_yaml_file = self.find_file(
            "cv/cameracalibration/chromatic_aberration/ca_photo_calib.yaml"
        )

        self.test_image = self.get_sample(
            "cv/cameracalibration/chromatic_aberration/ca_photo.png", 1
        )
        self.assertIsNotNone(self.test_image, "Failed to load test image")
        self.assertFalse(self.test_image.size == 0, "Failed to load test image")

    def test_chromatic_aberration_corrector_load_calibration(self):
        corrector = cv.ChromaticAberrationCorrector(self.test_yaml_file)
        self.assertIsNotNone(corrector)

        with self.assertRaises((cv.error, SystemError)):
            _ = cv.ChromaticAberrationCorrector("non_existent_file.yaml")

    def test_chromatic_aberration_correction(self):
        corrected = cv.correctChromaticAberration(self.test_image, self.test_yaml_file)

        self.assertEqual(corrected.shape[:2], self.test_image.shape[:2])
        self.assertEqual(corrected.dtype, self.test_image.dtype)

        diff = cv.absdiff(self.test_image, corrected)
        sum_diff = cv.sumElems(diff)
        self.assertGreater(sum(sum_diff[:3]), 0.0)

    def test_chromatic_aberration_correction_tablet(self):
        test_yaml_file = self.find_file(
            "cv/cameracalibration/chromatic_aberration/calib_result_tablet.yaml"
        )
        test_image = self.get_sample(
            "cv/cameracalibration/chromatic_aberration/tablet_circles_.png", 1
        )
        self.assertIsNotNone(test_image, "Failed to load test image")
        self.assertFalse(test_image.size == 0, "Failed to load test image")

        corrected = cv.correctChromaticAberration(test_image, test_yaml_file)

        self.assertEqual(corrected.shape[:2], test_image.shape[:2])
        self.assertEqual(corrected.dtype, test_image.dtype)

        diff = cv.absdiff(test_image, corrected)
        sum_diff = cv.sumElems(diff)
        self.assertGreater(sum(sum_diff[:3]), 0.0)

    def test_chromatic_aberration_correction_synthetic_simple_warp(self):
        test_yaml_file = self.find_file(
            "cv/cameracalibration/chromatic_aberration/simple_warp.yaml"
        )
        test_image = self.get_sample(
            "cv/cameracalibration/chromatic_aberration/synthetic_simple_warp.png", 1
        )
        self.assertIsNotNone(test_image, "Failed to load test image")
        self.assertFalse(test_image.size == 0, "Failed to load test image")

        corrected = cv.correctChromaticAberration(test_image, test_yaml_file)

        self.assertEqual(corrected.shape[:2], test_image.shape[:2])
        self.assertEqual(corrected.dtype, test_image.dtype)

        diff = cv.absdiff(test_image, corrected)
        sum_diff = cv.sumElems(diff)
        self.assertGreater(sum(sum_diff[:3]), 0.0)

    def test_chromatic_aberration_correction_synthetic_radial(self):
        test_yaml_file = self.find_file("cv/cameracalibration/chromatic_aberration/radial.yaml")
        test_image = self.get_sample(
            "cv/cameracalibration/chromatic_aberration/synthetic_radial.png", 1
        )
        self.assertIsNotNone(test_image, "Failed to load test image")
        self.assertFalse(test_image.size == 0, "Failed to load test image")

        corrected = cv.correctChromaticAberration(test_image, test_yaml_file)

        self.assertEqual(corrected.shape[:2], test_image.shape[:2])
        self.assertEqual(corrected.dtype, test_image.dtype)

        diff = cv.absdiff(test_image, corrected)
        sum_diff = cv.sumElems(diff)
        self.assertGreater(sum(sum_diff[:3]), 0.0)

    def test_chromatic_aberration_correction_invalid_input(self):
        gray = cv.cvtColor(self.test_image, cv.COLOR_BGR2GRAY)
        with self.assertRaises(cv.error):
            _ = cv.correctChromaticAberration(gray, self.test_yaml_file)

    def test_correct_chromatic_aberration_function(self):
        corrected = cv.correctChromaticAberration(self.test_image, self.test_yaml_file)

        self.assertEqual(corrected.shape[0], self.test_image.shape[0])
        self.assertEqual(corrected.shape[1], self.test_image.shape[1])
        self.assertEqual(corrected.dtype, self.test_image.dtype)

    def test_yaml_reading_integration(self):
        fs = cv.FileStorage(self.test_yaml_file, cv.FileStorage_READ)
        self.assertTrue(fs.isOpened())

        red_node = fs.getNode("red_channel")
        blue_node = fs.getNode("blue_channel")

        coeffs_x = red_node.getNode("coeffs_x")
        self.assertIsNotNone(coeffs_x)
        self.assertEqual(coeffs_x.size(), EXPECTED_COEFFS_SIZE)

        coeffs_x = blue_node.getNode("coeffs_x")
        self.assertIsNotNone(coeffs_x)
        self.assertEqual(coeffs_x.size(), EXPECTED_COEFFS_SIZE)

        coeffs_y = red_node.getNode("coeffs_y")
        self.assertIsNotNone(coeffs_y)
        self.assertEqual(coeffs_y.size(), EXPECTED_COEFFS_SIZE)

        coeffs_y = blue_node.getNode("coeffs_y")
        self.assertIsNotNone(coeffs_y)
        self.assertEqual(coeffs_y.size(), EXPECTED_COEFFS_SIZE)

        fs.release()

    def test_function_class_equivalence(self):
        if not hasattr(cv, "ChromaticAberrationCorrector"):
            self.skipTest("cv.ChromaticAberrationCorrector is not available in Python bindings")

        corrector = cv.ChromaticAberrationCorrector(self.test_yaml_file)
        ref = corrector.correctImage(self.test_image)
        out = cv.correctChromaticAberration(self.test_image, self.test_yaml_file)

        self.assertEqual(ref.shape[:2], self.test_image.shape[:2])
        self.assertEqual(ref.dtype, self.test_image.dtype)

        diff = cv.absdiff(ref, out)
        inf_norm = cv.norm(diff, cv.NORM_INF)
        self.assertLessEqual(inf_norm, 1)

        nz = cv.countNonZero(diff.reshape(-1, 1))
        self.assertEqual(nz, 0, msg=f"{nz} pixels differ between implementations")


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
