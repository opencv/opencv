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

    def test_load_calib_and_correct_image(self):
        coeffMat, degree, calibW, calibH = cv.loadCalibrationResultFromFile(self.test_yaml_file)

        self.assertIsInstance(coeffMat, np.ndarray)
        self.assertEqual(coeffMat.dtype, np.float32)
        self.assertEqual(coeffMat.shape[0], 4)
        self.assertGreater(coeffMat.shape[1], 0)
        self.assertGreater(degree, 0)
        self.assertGreater(calibW, 0)
        self.assertGreater(calibH, 0)
        self.assertEqual(coeffMat.shape[1], EXPECTED_COEFFS_SIZE)

        self.assertEqual(self.test_image.shape[1], calibW)
        self.assertEqual(self.test_image.shape[0], calibH)

        corrected = cv.correctChromaticAberration(self.test_image, coeffMat, calibW, calibH, degree)

        self.assertEqual(corrected.shape[:2], self.test_image.shape[:2])
        self.assertEqual(corrected.dtype, self.test_image.dtype)

        diff = cv.absdiff(self.test_image, corrected)
        sum_diff = cv.sumElems(diff)
        self.assertGreater(sum(sum_diff[:3]), 0.0)

    def test_yaml_contents_as_expected(self):
        fs = cv.FileStorage(self.test_yaml_file, cv.FileStorage_READ)
        self.assertTrue(fs.isOpened())

        red_node = fs.getNode("red_channel")
        blue_node = fs.getNode("blue_channel")
        self.assertTrue(red_node.isMap())
        self.assertTrue(blue_node.isMap())

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

    def test_invalid_single_channel(self):
        coeffMat, degree, calibW, calibH = self._load_calib()

        gray = cv.cvtColor(self.test_image, cv.COLOR_BGR2GRAY)
        with self.assertRaises(cv.error):
            _ = cv.correctChromaticAberration(gray, coeffMat, calibW, calibH, degree)

    def test_empty_coeff_mat(self):
        coeffMat, degree, calibW, calibH = self._load_calib()

        emptyCoeff = np.empty((0, 0), dtype=np.float32)
        with self.assertRaises(cv.error):
            _ = cv.correctChromaticAberration(self.test_image, emptyCoeff, calibW, calibH, degree)

    def test_mismatched_image_size(self):
        coeffMat, degree, calibW, calibH = self._load_calib()

        resized = cv.resize(self.test_image, (self.test_image.shape[1] // 2, self.test_image.shape[0] // 2))
        with self.assertRaises(cv.error):
            _ = cv.correctChromaticAberration(resized, coeffMat, calibW, calibH, degree)

    def test_wrong_coeff_type(self):
        coeffMat, degree, calibW, calibH = self._load_calib()

        wrongType = coeffMat.astype(np.float64)
        with self.assertRaises(cv.error):
            _ = cv.correctChromaticAberration(self.test_image, wrongType, calibW, calibH, degree)

    def test_degree_does_not_match_coeff_cols(self):
        coeffMat, degree, calibW, calibH = self._load_calib()

        wrongDegree = max(1, degree - 1)
        self.assertNotEqual(EXPECTED_COEFFS_SIZE, coeffMat.shape[1])
        with self.assertRaises(cv.error):
            _ = cv.correctChromaticAberration(self.test_image, coeffMat, calibW, calibH, wrongDegree)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
