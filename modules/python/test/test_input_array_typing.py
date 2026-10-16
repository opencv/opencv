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


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
