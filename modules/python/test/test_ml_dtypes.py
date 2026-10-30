#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv
try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

from tests_common import NewOpenCVTests

NON_CONVENTIONAL_DEPTHS = (cv.CV_16BF, cv.CV_8F_E4M3FN, cv.CV_8F_E4M3FNUZ)


class non_conventional_dtypes_test(NewOpenCVTests):
    # Runs with or without ml_dtypes.

    def setUp(self):
        super(non_conventional_dtypes_test, self).setUp()
        self.addCleanup(cv.setUseExactNumpyDtypes, cv.useExactNumpyDtypes())
        cv.setUseExactNumpyDtypes(False)

    def test_flag_roundtrip(self):
        cv.setUseExactNumpyDtypes(True)
        self.assertTrue(cv.useExactNumpyDtypes())
        cv.setUseExactNumpyDtypes(False)
        self.assertFalse(cv.useExactNumpyDtypes())

    def test_disabled_widens_to_float32(self):
        for depth in NON_CONVENTIONAL_DEPTHS:
            for m in cv.utils.generateVectorOfMat(1, 3, 4, depth):
                self.assertEqual(m.dtype, np.float32)

    def test_enabled_without_ml_dtypes_widens_to_float32(self):
        if ml_dtypes is not None:
            self.skipTest("ml_dtypes is installed")
        cv.setUseExactNumpyDtypes(True)
        for depth in NON_CONVENTIONAL_DEPTHS:
            for m in cv.utils.generateVectorOfMat(1, 3, 4, depth):
                self.assertEqual(m.dtype, np.float32)

    def test_short_bool_array_scalar_operands(self):
        for channels in range(1, 5):
            src = np.zeros((2, 3, channels), dtype=np.float32)
            operand = np.array([True, False, True, True][:channels])
            np.testing.assert_array_equal(cv.add(src, operand),
                                          cv.add(src, tuple(float(v) for v in operand)))


class ml_dtypes_test(NewOpenCVTests):

    def setUp(self):
        super(ml_dtypes_test, self).setUp()
        if ml_dtypes is None:
            self.skipTest("ml_dtypes is not installed")
        self.addCleanup(cv.setUseExactNumpyDtypes, cv.useExactNumpyDtypes())
        cv.setUseExactNumpyDtypes(True)
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

    def test_inputs_accepted_with_outputs_disabled(self):
        cv.setUseExactNumpyDtypes(False)
        for dt, _, name, _ in self.cases:
            a = np.arange(6, dtype=np.float32).reshape(2, 3).astype(dt)
            self.assertIn("type(-1)=%sC1" % name, cv.utils.dumpInputArray(a))
            result = cv.flip(a, 1)
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(result, a[:, ::-1].astype(np.float32))
            # The caller's own dst keeps its dtype and is returned as is.
            d = np.empty_like(a)
            self.assertIs(cv.flip(a, 1, dst=d), d)
            self.assertSameBits(d, a[:, ::-1])

    def test_short_array_scalar_operands(self):
        # A float32 src keeps the result float32, so the operand is checked with the flag off too.
        for exact in (False, True):
            cv.setUseExactNumpyDtypes(exact)
            for dt, _, _, _ in self.cases:
                for channels in range(1, 5):
                    src = np.zeros((2, 3, channels), dtype=np.float32)
                    values = np.arange(channels, dtype=np.float32) - 1
                    operand = values.astype(dt)
                    expected = cv.add(src, tuple(float(v) for v in values))
                    np.testing.assert_array_equal(cv.add(src, operand), expected)
                    np.testing.assert_array_equal(cv.add(src, operand[::-1]),
                                                  cv.add(src, tuple(float(v) for v in values[::-1])))
        img = np.zeros((2, 3, 3), dtype=ml_dtypes.bfloat16)
        operand = np.array([1, 2, 3], dtype=ml_dtypes.bfloat16)
        self.assertSameBits(cv.add(img, operand), img + operand)

    def test_roundtrip_through_numpy_allocator(self):
        # 240 is the largest finite float8_e4m3fnuz value, so every case keeps these values exactly.
        src = np.array([[0.5, -1.25, 3.0, 7.0], [240.0, -240.0, 0.0, -0.5]], dtype=np.float32)
        for dt, _, _, _ in self.cases:
            a = src.astype(dt)
            np.testing.assert_array_equal(a.astype(np.float32), src)
            self.assertSameBits(cv.flip(a, 1), a[:, ::-1])
            self.assertSameBits(cv.transpose(a), a.T)
            strided = a[:, ::2]
            self.assertSameBits(cv.flip(strided, 0), strided[::-1])

    def test_dst_is_returned(self):
        for dt, _, _, _ in self.cases:
            a = np.arange(6, dtype=np.float32).reshape(2, 3).astype(dt)
            d = np.empty_like(a)
            self.assertIs(cv.flip(a, 1, dst=d), d)

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
