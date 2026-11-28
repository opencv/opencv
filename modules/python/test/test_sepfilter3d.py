#!/usr/bin/env python

import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests


class SepFilter3DTest(NewOpenCVTests):
    def test_volume_shape_and_channels(self):
        for dtype in (np.uint8, np.float16, np.float32):
            for shape in ((3, 4, 5), (3, 4, 5, 1), (3, 4, 5, 3)):
                with self.subTest(dtype=dtype, shape=shape):
                    volume = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
                    identity = np.array([1], np.float32)
                    next_sample = np.array([0, 0, 1], np.float32)
                    expected = volume[[1, 2, 1]][:, :, [1, 2, 3, 4, 3]]

                    result = cv.sepFilter3D(volume, next_sample, identity, next_sample)

                    self.assertEqual(result.dtype, dtype)
                    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    NewOpenCVTests.bootstrap()
