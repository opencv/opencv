#!/usr/bin/env python
from __future__ import print_function

import ctypes
from functools import partial

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests, unittest


def is_numeric(dtype):
    return np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)


def get_limits(dtype):
    if not is_numeric(dtype):
        return None, None

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    else:
        info = np.finfo(dtype)
    return info.min, info.max


def get_conversion_error_msg(value, expected, actual):
    return 'Conversion "{}" of type "{}" failed\nExpected: "{}" vs Actual "{}"'.format(
        value, type(value).__name__, expected, actual
    )


def get_no_exception_msg(value):
    return 'Exception is not risen for {} of type {}'.format(value, type(value).__name__)

class Bindings(NewOpenCVTests):

    def test_inheritance(self):
        bm = cv.StereoBM_create()
        bm.getPreFilterCap()  # from StereoBM
        bm.getBlockSize()  # from SteroMatcher

        boost = cv.ml.Boost_create()
        boost.getBoostType()  # from ml::Boost
        boost.getMaxDepth()  # from ml::DTrees
        boost.isClassifier()  # from ml::StatModel

    def test_redirectError(self):
        try:
            cv.imshow("", None)  # This causes an assert
            self.assertEqual("Dead code", 0)
        except cv.error as _e:
            pass

        handler_called = [False]

        def test_error_handler(status, func_name, err_msg, file_name, line):
            handler_called[0] = True

        cv.redirectError(test_error_handler)
        try:
            cv.imshow("", None)  # This causes an assert
            self.assertEqual("Dead code", 0)
        except cv.error as _e:
            self.assertEqual(handler_called[0], True)
            pass

        cv.redirectError(None)
        try:
            cv.imshow("", None)  # This causes an assert
            self.assertEqual("Dead code", 0)
        except cv.error as _e:
            pass


class Arguments(NewOpenCVTests):

    def _try_to_convert(self, conversion, value):
        try:
            result = conversion(value).lower()
        except Exception as e:
            self.fail(
                '{} "{}" is risen for conversion {} of type {}'.format(
                    type(e).__name__, e, value, type(value).__name__
                )
            )
        else:
            return result

    def test_InputArray(self):
        res1 = cv.utils.dumpInputArray(None)
        # self.assertEqual(res1, "InputArray: noArray()")  # not supported
        self.assertEqual(res1, "InputArray: empty()=true kind=0x00010000 flags=0x01010000 total(-1)=0 dims(-1)=0 size(-1)=0x0 type(-1)=CV_8UC1")
        res2_1 = cv.utils.dumpInputArray((1, 2))
        self.assertEqual(res2_1, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=2 dims(-1)=2 size(-1)=1x2 type(-1)=CV_64FC1")
        res2_2 = cv.utils.dumpInputArray(1.5)  # Scalar(1.5, 1.5, 1.5, 1.5)
        self.assertEqual(res2_2, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=4 dims(-1)=2 size(-1)=1x4 type(-1)=CV_64FC1")
        a = np.array([[1, 2], [3, 4], [5, 6]])
        res3 = cv.utils.dumpInputArray(a)  # 32SC1
        self.assertEqual(res3, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=6 dims(-1)=2 size(-1)=2x3 type(-1)=CV_32SC1")
        a = np.array([[[1, 2], [3, 4], [5, 6]]], dtype='f')
        res4 = cv.utils.dumpInputArray(a)  # 32FC2
        self.assertEqual(res4, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=3 dims(-1)=2 size(-1)=3x1 type(-1)=CV_32FC2")
        a = np.array([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=float)
        res5 = cv.utils.dumpInputArray(a)  # 64FC2
        self.assertEqual(res5, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=3 dims(-1)=2 size(-1)=1x3 type(-1)=CV_64FC2")

    def test_InputArrayOfArrays(self):
        res1 = cv.utils.dumpInputArrayOfArrays(None)
        # self.assertEqual(res1, "InputArray: noArray()")  # not supported
        self.assertEqual(res1, "InputArrayOfArrays: empty()=true kind=0x00050000 flags=0x01050000 total(-1)=0 dims(-1)=1 size(-1)=0x0")
        res2_1 = cv.utils.dumpInputArrayOfArrays((1, 2))  # { Scalar:all(1), Scalar::all(2) }
        self.assertEqual(res2_1, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=2 dims(-1)=1 size(-1)=2x1 type(0)=CV_64FC1 dims(0)=2 size(0)=1x4 type(0)=CV_64FC1")
        res2_2 = cv.utils.dumpInputArrayOfArrays([1.5])
        self.assertEqual(res2_2, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=1 dims(-1)=1 size(-1)=1x1 type(0)=CV_64FC1 dims(0)=2 size(0)=1x4 type(0)=CV_64FC1")
        a = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        res3 = cv.utils.dumpInputArrayOfArrays([a, b])
        self.assertEqual(res3, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=2 dims(-1)=1 size(-1)=2x1 type(0)=CV_32SC1 dims(0)=2 size(0)=2x3 type(0)=CV_32SC1")
        c = np.array([[[1, 2], [3, 4], [5, 6]]], dtype='f')
        res4 = cv.utils.dumpInputArrayOfArrays([c, a, b])
        self.assertEqual(res4, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=3 dims(-1)=1 size(-1)=3x1 type(0)=CV_32FC2 dims(0)=2 size(0)=3x1 type(0)=CV_32FC2")

    def test_parse_to_bool_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpBool)
        for convertible_true in (True, 1, 64, np.bool(1), np.int8(123), np.int16(11), np.int32(2),
                                 np.int64(1), np.bool_(3), np.bool8(12)):
            actual = try_to_convert(convertible_true)
            self.assertEqual('bool: true', actual,
                             msg=get_conversion_error_msg(convertible_true, 'bool: true', actual))

        for convertible_false in (False, 0, np.uint8(0), np.bool_(0), np.int_(0)):
            actual = try_to_convert(convertible_false)
            self.assertEqual('bool: false', actual,
                             msg=get_conversion_error_msg(convertible_false, 'bool: false', actual))

    def test_parse_to_bool_not_convertible(self):
        for not_convertible in (1.2, np.float(2.3), 's', 'str', (1, 2), [1, 2], complex(1, 1),
                                complex(imag=2), complex(1.1), np.array([1, 0], dtype=np.bool)):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpBool(not_convertible)

    def test_parse_to_bool_convertible_extra(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpBool)
        _, max_size_t = get_limits(ctypes.c_size_t)
        for convertible_true in (-1, max_size_t):
            actual = try_to_convert(convertible_true)
            self.assertEqual('bool: true', actual,
                             msg=get_conversion_error_msg(convertible_true, 'bool: true', actual))

    def test_parse_to_bool_not_convertible_extra(self):
        for not_convertible in (np.array([False]), np.array([True], dtype=np.bool)):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpBool(not_convertible)

    def test_parse_to_int_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpInt)
        min_int, max_int = get_limits(ctypes.c_int)
        for convertible in (-10, -1, 2, int(43.2), np.uint8(15), np.int8(33), np.int16(-13),
                            np.int32(4), np.int64(345), (23), min_int, max_int, np.int_(33)):
            expected = 'int: {0:d}'.format(convertible)
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_int_not_convertible(self):
        min_int, max_int = get_limits(ctypes.c_int)
        for not_convertible in (1.2, np.float(4), float(3), np.double(45), 's', 'str',
                                np.array([1, 2]), (1,), [1, 2], min_int - 1, max_int + 1,
                                complex(1, 1), complex(imag=2), complex(1.1)):
            with self.assertRaises((TypeError, OverflowError, ValueError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpInt(not_convertible)

    def test_parse_to_int_not_convertible_extra(self):
        for not_convertible in (np.bool_(True), True, False, np.float32(2.3),
                                np.array([3, ], dtype=int), np.array([-2, ], dtype=np.int32),
                                np.array([1, ], dtype=np.int), np.array([11, ], dtype=np.uint8)):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpInt(not_convertible)

    def test_parse_to_size_t_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpSizeT)
        _, max_uint = get_limits(ctypes.c_uint)
        for convertible in (2, max_uint, (12), np.uint8(34), np.int8(12), np.int16(23),
                            np.int32(123), np.int64(344), np.uint64(3), np.uint16(2), np.uint32(5),
                            np.uint(44)):
            expected = 'size_t: {0:d}'.format(convertible).lower()
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_size_t_not_convertible(self):
        min_long, _ = get_limits(ctypes.c_long)
        for not_convertible in (1.2, True, False, np.bool_(True), np.float(4), float(3),
                                np.double(45), 's', 'str', np.array([1, 2]), (1,), [1, 2],
                                np.float64(6), complex(1, 1), complex(imag=2), complex(1.1),
                                -1, min_long, np.int8(-35)):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpSizeT(not_convertible)

    def test_parse_to_size_t_convertible_extra(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpSizeT)
        _, max_size_t = get_limits(ctypes.c_size_t)
        for convertible in (max_size_t,):
            expected = 'size_t: {0:d}'.format(convertible).lower()
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_size_t_not_convertible_extra(self):
        for not_convertible in (np.bool_(True), True, False, np.array([123, ], dtype=np.uint8),):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpSizeT(not_convertible)

    def test_parse_to_float_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpFloat)
        min_float, max_float = get_limits(ctypes.c_float)
        for convertible in (2, -13, 1.24, float(32), np.float(32.45), np.double(12.23),
                            np.float32(-12.3), np.float64(3.22), np.float_(-1.5), min_float,
                            max_float, np.inf, -np.inf, float('Inf'), -float('Inf'),
                            np.double(np.inf), np.double(-np.inf), np.double(float('Inf')),
                            np.double(-float('Inf'))):
            expected = 'Float: {0:.2f}'.format(convertible).lower()
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

        # Workaround for Windows NaN tests due to Visual C runtime
        # special floating point values (indefinite NaN)
        for nan in (float('NaN'), np.nan, np.float32(np.nan), np.double(np.nan),
                    np.double(float('NaN'))):
            actual = try_to_convert(nan)
            self.assertIn('nan', actual, msg="Can't convert nan of type {} to float. "
                          "Actual: {}".format(type(nan).__name__, actual))

        min_double, max_double = get_limits(ctypes.c_double)
        for inf in (min_float * 10, max_float * 10, min_double, max_double):
            expected = 'float: {}inf'.format('-' if inf < 0 else '')
            actual = try_to_convert(inf)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(inf, expected, actual))

    def test_parse_to_float_not_convertible(self):
        for not_convertible in ('s', 'str', (12,), [1, 2], np.array([1, 2], dtype=np.float),
                                np.array([1, 2], dtype=np.double), complex(1, 1), complex(imag=2),
                                complex(1.1)):
            with self.assertRaises((TypeError), msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpFloat(not_convertible)

    def test_parse_to_float_not_convertible_extra(self):
        for not_convertible in (np.bool_(False), True, False, np.array([123, ], dtype=int),
                                np.array([1., ]), np.array([False]),
                                np.array([True], dtype=np.bool)):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpFloat(not_convertible)

    def test_parse_to_double_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpDouble)
        min_float, max_float = get_limits(ctypes.c_float)
        min_double, max_double = get_limits(ctypes.c_double)
        for convertible in (2, -13, 1.24, np.float(32.45), float(2), np.double(12.23),
                            np.float32(-12.3), np.float64(3.22), np.float_(-1.5), min_float,
                            max_float, min_double, max_double, np.inf, -np.inf, float('Inf'),
                            -float('Inf'), np.double(np.inf), np.double(-np.inf),
                            np.double(float('Inf')), np.double(-float('Inf'))):
            expected = 'Double: {0:.2f}'.format(convertible).lower()
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

        # Workaround for Windows NaN tests due to Visual C runtime
        # special floating point values (indefinite NaN)
        for nan in (float('NaN'), np.nan, np.double(np.nan),
                    np.double(float('NaN'))):
            actual = try_to_convert(nan)
            self.assertIn('nan', actual, msg="Can't convert nan of type {} to double. "
                          "Actual: {}".format(type(nan).__name__, actual))

    def test_parse_to_double_not_convertible(self):
        for not_convertible in ('s', 'str', (12,), [1, 2], np.array([1, 2], dtype=np.float),
                                np.array([1, 2], dtype=np.double), complex(1, 1), complex(imag=2),
                                complex(1.1)):
            with self.assertRaises((TypeError), msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpDouble(not_convertible)

    def test_parse_to_double_not_convertible_extra(self):
        for not_convertible in (np.bool_(False), True, False, np.array([123, ], dtype=int),
                                np.array([1., ]), np.array([False]),
                                np.array([12.4], dtype=np.double), np.array([True], dtype=np.bool)):
            with self.assertRaises((TypeError, OverflowError),
                                   msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpDouble(not_convertible)

    def test_parse_to_cstring_convertible(self):
        try_to_convert = partial(self._try_to_convert, cv.utils.dumpCString)
        for convertible in ('s', 'str', str(123), ('char'), np.str('test1'), np.str_('test2')):
            expected = 'string: ' + convertible
            actual = try_to_convert(convertible)
            self.assertEqual(expected, actual,
                             msg=get_conversion_error_msg(convertible, expected, actual))

    def test_parse_to_cstring_not_convertible(self):
        for not_convertible in ((12,), ('t', 'e', 's', 't'), np.array(['123', ]),
                                np.array(['t', 'e', 's', 't']), 1, -1.4, True, False, None):
            with self.assertRaises((TypeError), msg=get_no_exception_msg(not_convertible)):
                _ = cv.utils.dumpCString(not_convertible)


class SamplesFindFile(NewOpenCVTests):

    def test_ExistedFile(self):
        res = cv.samples.findFile('lena.jpg', False)
        self.assertNotEqual(res, '')

    def test_MissingFile(self):
        res = cv.samples.findFile('non_existed.file', False)
        self.assertEqual(res, '')

    def test_MissingFileException(self):
        try:
            _res = cv.samples.findFile('non_existed.file', True)
            self.assertEqual("Dead code", 0)
        except cv.error as _e:
            pass


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
