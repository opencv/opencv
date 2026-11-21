#!/usr/bin/env python

import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests


class copy_make_border_test(NewOpenCVTests):

    def test_copy_make_border_with_values(self):
        image = np.arange(3 * 4 * 9, dtype=np.uint8).reshape(3, 4, 9)
        values = (-20.5, 1.25, 2.75, 3., 4., 5., 6., 254.75, 300.)
        backing = np.empty((7, 8, 9), dtype=np.uint8)
        dst = backing[1:-1, 1:-1]

        expected = np.empty_like(dst)
        expected[...] = [0, 1, 3, 3, 4, 5, 6, 255, 255]
        expected[1:-1, 1:-1] = image
        for border_values in (values, np.array(values)):
            with self.subTest(type=type(border_values)):
                dst.fill(231)
                result = cv.copyMakeBorderWithBorderValues(image, 1, 1, 1, 1,
                                                           cv.BORDER_CONSTANT, border_values, dst=dst)
                self.assertIs(result, dst)
                np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
