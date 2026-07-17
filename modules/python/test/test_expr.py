#!/usr/bin/env python
'''
cv.texpr() - the string front-end of the broadcasting element-wise expression engine
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests


class texpr_test(NewOpenCVTests):

    def test_basic_arith(self):
        a = np.random.uniform(1, 10, (12, 15)).astype(np.float32)
        b = np.random.uniform(1, 10, (12, 15)).astype(np.float32)
        res = cv.texpr("{0} * 2.5 + {1}", [a, b])
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 1)
        self.assertLessEqual(np.max(np.abs(res[0] - (a * 2.5 + b))), 1e-3)
        # the idiomatic single-result unpacking
        r, = cv.texpr("{0} * 2.5 + {1}", [a, b])
        self.assertTrue(np.array_equal(r, res[0]))

    def test_fused_absdiff(self):
        a = np.random.randint(0, 255, (20, 30)).astype(np.uint8)
        b = np.random.randint(0, 255, (20, 30)).astype(np.uint8)
        got = cv.texpr("abs({0} - {1})", [a, b])[0]
        self.assertEqual(got.dtype, np.uint8)
        self.assertTrue(np.array_equal(got, cv.absdiff(a, b)))

    def test_type_cast(self):
        a = np.random.uniform(-50, 300, (10, 10)).astype(np.float32)
        got = cv.texpr("uint8({0})", [a])[0]
        self.assertEqual(got.dtype, np.uint8)
        ref = np.clip(np.rint(a), 0, 255).astype(np.uint8)
        self.assertTrue(np.array_equal(got, ref))

    def test_broadcasting(self):
        img = np.random.uniform(0, 255, (8, 6, 3)).astype(np.float32)
        row = np.random.uniform(1, 2, (1, 6, 3)).astype(np.float32)
        got = cv.texpr("{0} * {1}", [img, row])[0]
        self.assertLessEqual(np.max(np.abs(got - img * row)), 1e-3)

    def test_ternary_and_compare(self):
        a = np.random.uniform(0, 100, (9, 14)).astype(np.float32)
        b = np.random.uniform(0, 100, (9, 14)).astype(np.float32)
        got = cv.texpr("{0} > {1} ? {0} : {1}", [a, b])[0]
        self.assertTrue(np.array_equal(got, np.maximum(a, b)))

    def test_pow_operator(self):
        a = np.random.uniform(0.5, 2, (7, 11)).astype(np.float32)
        got = cv.texpr("3 * {0} ** 2", [a])[0]     # ** binds tighter than *
        self.assertLessEqual(np.max(np.abs(got - 3 * a ** 2) / (3 * a ** 2)), 1e-5)

    def test_math_functions(self):
        x = np.random.uniform(0.05, 9, (13, 17)).astype(np.float32)
        got = cv.texpr("exp(-{0}) + log({0}) + sqrt({0})", [x])[0]
        ref = np.exp(-x.astype(np.float64)) + np.log(x.astype(np.float64)) + np.sqrt(x.astype(np.float64))
        self.assertLessEqual(np.max(np.abs(got - ref)), 1e-4)

    def test_clamp_scalar_literals(self):
        a = np.random.uniform(-100, 355, (10, 21)).astype(np.float32)
        got = cv.texpr("clamp({0}, 10, 200)", [a])[0]
        self.assertTrue(np.array_equal(got, np.clip(a, 10, 200)))

    def test_named_temporary(self):
        a = np.random.uniform(1, 10, (6, 8)).astype(np.float32)
        b = np.random.uniform(1, 10, (6, 8)).astype(np.float32)
        got = cv.texpr("d = {0} - {1}; d*d", [a, b])[0]
        self.assertLessEqual(np.max(np.abs(got - (a - b) ** 2)), 1e-3)

    def test_tuple_outputs(self):
        a = np.random.uniform(1, 10, (5, 9)).astype(np.float32)
        b = np.random.uniform(1, 10, (5, 9)).astype(np.float32)
        res = cv.texpr("({0} + {1}, {0} - {1})", [a, b])
        self.assertEqual(len(res), 2)
        self.assertLessEqual(np.max(np.abs(res[0] - (a + b))), 1e-3)
        self.assertLessEqual(np.max(np.abs(res[1] - (a - b))), 1e-3)

    def test_cart_to_polar(self):
        x = np.random.uniform(-5, 5, (11, 13)).astype(np.float32)
        y = np.random.uniform(-5, 5, (11, 13)).astype(np.float32)
        mag, ang = cv.texpr("(hypot({0},{1}), atan2({1},{0}))", [x, y])
        self.assertLessEqual(np.max(np.abs(mag - np.hypot(x, y))), 1e-3)
        self.assertLessEqual(np.max(np.abs(ang - np.arctan2(y, x))), 2e-4)

    def test_int_saturation(self):
        # 32-bit add/subtract saturate (the new v_add_sat kernels)
        a = np.array([[2**31 - 1, -2**31, 0]], dtype=np.int32)
        b = np.array([[1, -1, -2**31]], dtype=np.int32)
        s = cv.texpr("{0} + {1}", [a, b])[0]
        d = cv.texpr("{0} - {1}", [a, b])[0]
        self.assertEqual(s[0, 0], 2**31 - 1)   # MAX + 1 -> MAX
        self.assertEqual(s[0, 1], -2**31)      # MIN - 1 -> MIN
        self.assertEqual(d[0, 2], 2**31 - 1)   # 0 - MIN -> MAX


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
