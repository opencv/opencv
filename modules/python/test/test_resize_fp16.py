#!/usr/bin/env python

import unittest

import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests


class resize_fp16_test(NewOpenCVTests):
    def test_matches_float32_working_image(self):
        rng = np.random.default_rng(30216)
        for channels in (1, 3, 4, 9):
            storage = rng.uniform(-1, 1, (19, 29, channels)).astype(np.float16)
            src = storage[1:18, 2:27]
            for mode in (cv.INTER_LINEAR, cv.INTER_LINEAR_EXACT,
                         cv.INTER_CUBIC, cv.INTER_LANCZOS4):
                for size in ((1, 1), (13, 9), (43, 31)):
                    expected = cv.resize(src.astype(np.float32), size,
                                         interpolation=mode).astype(np.float16)
                    actual = cv.resize(src, size, interpolation=mode)
                    self.assertEqual(actual.dtype, np.float16)
                    np.testing.assert_allclose(actual, expected,
                                               rtol=2e-3, atol=2e-5)

    def test_nearest_preserves_half_bits(self):
        bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        src = bits.view(np.float16)
        expected = np.repeat(np.repeat(bits, 2, axis=0), 2, axis=1)
        for mode in (cv.INTER_NEAREST, cv.INTER_NEAREST_EXACT):
            actual = cv.resize(src, (512, 512), interpolation=mode)
            np.testing.assert_array_equal(actual.view(np.uint16), expected)

    def test_area_and_preallocated_destination(self):
        src = np.full((24, 32, 3), 60000, dtype=np.float16)
        for size in ((16, 12), (11, 7)):
            dst = np.empty((size[1], size[0], 3), dtype=np.float16)
            result = cv.resize(src, size, dst=dst, interpolation=cv.INTER_AREA)
            self.assertTrue(np.shares_memory(result, dst))
            np.testing.assert_array_equal(dst, np.full_like(dst, 60000))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
