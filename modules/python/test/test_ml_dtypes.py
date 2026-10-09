#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv
try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

from tests_common import NewOpenCVTests


class ml_dtypes_test(NewOpenCVTests):

    def setUp(self):
        super(ml_dtypes_test, self).setUp()
        if ml_dtypes is None:
            self.skipTest("ml_dtypes is not installed")
        # (numpy dtype, OpenCV depth, typeToString prefix, relative half-ULP)
        self.cases = [
            (ml_dtypes.bfloat16, cv.CV_16BF, "CV_16BF", 2.0 ** -8),
            (ml_dtypes.float8_e4m3fn, cv.CV_8F_E4M3FN, "CV_8F", 2.0 ** -4),
            (ml_dtypes.float8_e4m3fnuz, cv.CV_8F_E4M3FNUZ, "CV_8FNUZ", 2.0 ** -4),
        ]

    def assertSameBits(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.array_equal(np.ascontiguousarray(actual).view(np.uint8),
                                       np.ascontiguousarray(expected).view(np.uint8)))

    def test_numpy_to_mat_keeps_exact_type(self):
        for dt, _, name, _ in self.cases:
            a = np.arange(6, dtype=np.float32).reshape(2, 3).astype(dt)
            self.assertIn("type(-1)=%sC1" % name, cv.utils.dumpInputArray(a))
            self.assertIn("type(-1)=%sC3" % name, cv.utils.dumpInputArray(np.stack([a, a, a], -1)))
            self.assertIn("type(-1)=%sC1" % name, cv.utils.dumpInputArray(np.array(1.5, dtype=dt)))

    def test_roundtrip_through_numpy_allocator(self):
        src = np.array([[0.5, -1.25, 3.0, 7.0], [448.0, -448.0, 0.0, -0.5]], dtype=np.float32)
        for dt, _, _, _ in self.cases:
            a = src.astype(dt)
            self.assertSameBits(cv.flip(a, 1), a[:, ::-1])
            self.assertSameBits(cv.transpose(a), a.T)
            strided = a[:, ::2]
            self.assertSameBits(cv.flip(strided, 0), strided[::-1])

    def test_mat_allocated_in_cpp_to_numpy(self):
        ref = cv.utils.generateVectorOfMat(2, 3, 4, cv.CV_32F)
        for dt, depth, _, rtol in self.cases:
            mats = cv.utils.generateVectorOfMat(2, 3, 4, depth)
            self.assertEqual(len(mats), len(ref))
            for m, r in zip(mats, ref):
                self.assertEqual(m.dtype, dt)
                self.assertEqual(m.shape, (3, 4))
                np.testing.assert_allclose(m.astype(np.float32), r, rtol=rtol, atol=2.0 ** -9)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
