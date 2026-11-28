#!/usr/bin/env python

import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests


class input_array_typing_test(NewOpenCVTests):
    def test_scalar_operands(self):
        image = np.ones((3, 5, 3), dtype=np.float32)
        scalar = (1.0, 2.0, 3.0, 0.0)
        added = cv.add(image, scalar)
        multiplied = cv.multiply(image, scalar)
        np.testing.assert_array_equal(added, image + np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(multiplied, image * np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_empty_second_remap_map(self):
        image = np.arange(15, dtype=np.uint8).reshape(3, 5)
        x, y = np.meshgrid(np.arange(5, dtype=np.float32), np.arange(3, dtype=np.float32))
        map1, _ = cv.convertMaps(x, y, cv.CV_16SC2, nninterpolation=True)
        remapped = cv.remap(image, map1, None, cv.INTER_NEAREST)
        np.testing.assert_array_equal(remapped, image)

    def test_small_array_with_scalar_issue_31045(self):
        # Regression test for Issue #31045:
        # A 1x1 or 0D array combined with a scalar operand must return (1, 1), not (4, 1).
        for dtype in [np.float64, np.float32, np.int32, np.uint8]:
            a = np.full((1, 1), 5, dtype=dtype)

            # cv.add
            res_add = cv.add(a, 1.0)
            self.assertEqual(res_add.shape, (1, 1))
            self.assertEqual(res_add[0, 0], 6)

            # cv.subtract
            res_sub = cv.subtract(a, 2.0)
            self.assertEqual(res_sub.shape, (1, 1))
            self.assertEqual(res_sub[0, 0], 3)

            # cv.multiply
            res_mul = cv.multiply(a, 2.0)
            self.assertEqual(res_mul.shape, (1, 1))
            self.assertEqual(res_mul[0, 0], 10)

            # cv.divide
            res_div = cv.divide(a, 2.0)
            self.assertEqual(res_div.shape, (1, 1))

            # cv.absdiff
            res_abs = cv.absdiff(a, 1.0)
            self.assertEqual(res_abs.shape, (1, 1))
            self.assertEqual(res_abs[0, 0], 4)

            # Python int operand
            res_int = cv.add(a, 1)
            self.assertEqual(res_int.shape, (1, 1))
            self.assertEqual(res_int[0, 0], 6)

            # Tuple operand
            res_tuple = cv.add(a, (1.0,))
            self.assertEqual(res_tuple.shape, (1, 1))
            self.assertEqual(res_tuple[0, 0], 6)

            # Scalar on left (scalar op array)
            res_rev_add = cv.add(1.0, a)
            self.assertEqual(res_rev_add.shape, (1, 1))
            self.assertEqual(res_rev_add[0, 0], 6)

            res_rev_sub = cv.subtract(10.0, a)
            self.assertEqual(res_rev_sub.shape, (1, 1))
            self.assertEqual(res_rev_sub[0, 0], 5)

        # 0D array (degenerate scalar array)
        a_0d = np.array(5.0, dtype=np.float64)
        res_0d = cv.add(a_0d, 1.0)
        self.assertEqual(res_0d.shape, (1, 1))
        self.assertEqual(res_0d[0, 0], 6.0)

        # 1D array with > 4 elements (1D arrays with <= 4 elements are treated as cv::Scalar per #24057)
        a_1d = np.full((5,), 5.0, dtype=np.float64)
        res_1d = cv.add(a_1d, 1.0)
        self.assertEqual(res_1d.shape, (5,))
        np.testing.assert_array_equal(res_1d, np.full((5,), 6.0, dtype=np.float64))

        # Legitimate 4x1 array with scalar operand must preserve (4, 1) shape
        a_4x1 = np.full((4, 1), 2.0, dtype=np.float64)
        res_4x1 = cv.add(a_4x1, 1.0)
        self.assertEqual(res_4x1.shape, (4, 1))
        np.testing.assert_array_equal(res_4x1, np.full((4, 1), 3.0, dtype=np.float64))

        # Array op array of shape (1, 1)
        b_1x1 = np.full((1, 1), 3.0, dtype=np.float64)
        res_arr = cv.add(np.full((1, 1), 2.0, dtype=np.float64), b_1x1)
        self.assertEqual(res_arr.shape, (1, 1))
        self.assertEqual(res_arr[0, 0], 5.0)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
