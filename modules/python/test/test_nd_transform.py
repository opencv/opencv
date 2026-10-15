#!/usr/bin/env python
'''
Binding coverage for the n-dimensional transpose, split, repeat, concat, slice and flipND,
including the overloads that share a name with the older 2D functions.
'''

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests


class nd_transform_test(NewOpenCVTests):

    def nd(self, shape, dtype=np.float32):
        return cv.Mat(np.arange(np.prod(shape), dtype=dtype).reshape(shape), wrap_channels=False)

    def test_transpose(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.assertEqual(cv.norm(cv.transpose(a), a.T, cv.NORM_INF), 0)

        b = self.nd((2, 3, 4))
        expected = np.transpose(b, (2, 0, 1))
        np.testing.assert_array_equal(cv.transpose(b, [2, 0, 1]), expected)
        np.testing.assert_array_equal(cv.transposeND(b, [2, 0, 1]), expected)

    def test_split(self):
        img = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        channels = cv.split(img)
        self.assertEqual(len(channels), 3)
        for c in range(3):
            np.testing.assert_array_equal(channels[c], img[:, :, c])

        b = self.nd((2, 6, 4))
        parts = cv.split(b, 1, [2, 4])
        self.assertEqual(len(parts), 2)
        np.testing.assert_array_equal(parts[0], b[:, :2])
        np.testing.assert_array_equal(parts[1], b[:, 2:])
        np.testing.assert_array_equal(cv.concat(parts, 1), b)

    def test_repeat(self):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_array_equal(cv.repeat(a, 2, 3), np.tile(a, (2, 3)))

        b = self.nd((2, 3, 4))
        np.testing.assert_array_equal(cv.repeat(b, [1, 2, 3]), np.tile(b, (1, 2, 3)))

    def test_slice(self):
        b = self.nd((4, 9, 5))
        np.testing.assert_array_equal(cv.slice(b, [0, 8], [4, -1], [1, -1]), b[:, ::-1])

    def test_flip(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        np.testing.assert_array_equal(cv.flip(a, -1), a[::-1, ::-1])
        np.testing.assert_array_equal(cv.flip(a, 0), a[::-1])
        np.testing.assert_array_equal(cv.flip(a, 1), a[:, ::-1])
        np.testing.assert_array_equal(cv.flipND(a, -1), a[:, ::-1])

        b = self.nd((2, 3, 4))
        np.testing.assert_array_equal(cv.flipND(b, 0), b[::-1])
        np.testing.assert_array_equal(cv.flipND(b, -1), b[..., ::-1])


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
