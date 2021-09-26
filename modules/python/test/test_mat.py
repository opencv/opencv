#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

import os
import sys
import unittest

from tests_common import NewOpenCVTests

try:
    if sys.version_info[:2] < (3, 0):
        raise unittest.SkipTest('Python 2.x is not supported')


    class MatTest(NewOpenCVTests):

        def test_mat_construct(self):
            data = np.random.random([10, 10, 3])

            #print(np.ndarray.__dictoffset__)  # 0
            #print(cv.Mat.__dictoffset__)  # 88 (> 0)
            #print(cv.Mat)  # <class cv2.Mat>
            #print(cv.Mat.__base__)  # <class 'numpy.ndarray'>

            mat_data0 = cv.Mat(data)
            assert isinstance(mat_data0, cv.Mat)
            assert isinstance(mat_data0, np.ndarray)
            self.assertEqual(mat_data0.wrap_channels, False)
            res0 = cv.utils.dumpInputArray(mat_data0)
            self.assertEqual(res0, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=300 dims(-1)=3 size(-1)=[10 10 3] type(-1)=CV_64FC1")

            mat_data1 = cv.Mat(data, wrap_channels=True)
            assert isinstance(mat_data1, cv.Mat)
            assert isinstance(mat_data1, np.ndarray)
            self.assertEqual(mat_data1.wrap_channels, True)
            res1 = cv.utils.dumpInputArray(mat_data1)
            self.assertEqual(res1, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=100 dims(-1)=2 size(-1)=10x10 type(-1)=CV_64FC3")

            mat_data2 = cv.Mat(mat_data1)
            assert isinstance(mat_data2, cv.Mat)
            assert isinstance(mat_data2, np.ndarray)
            self.assertEqual(mat_data2.wrap_channels, True)  # fail if __array_finalize__ doesn't work
            res2 = cv.utils.dumpInputArray(mat_data2)
            self.assertEqual(res2, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=100 dims(-1)=2 size(-1)=10x10 type(-1)=CV_64FC3")


        def test_mat_construct_4d(self):
            data = np.random.random([5, 10, 10, 3])

            mat_data0 = cv.Mat(data)
            assert isinstance(mat_data0, cv.Mat)
            assert isinstance(mat_data0, np.ndarray)
            self.assertEqual(mat_data0.wrap_channels, False)
            res0 = cv.utils.dumpInputArray(mat_data0)
            self.assertEqual(res0, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=1500 dims(-1)=4 size(-1)=[5 10 10 3] type(-1)=CV_64FC1")

            mat_data1 = cv.Mat(data, wrap_channels=True)
            assert isinstance(mat_data1, cv.Mat)
            assert isinstance(mat_data1, np.ndarray)
            self.assertEqual(mat_data1.wrap_channels, True)
            res1 = cv.utils.dumpInputArray(mat_data1)
            self.assertEqual(res1, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=500 dims(-1)=3 size(-1)=[5 10 10] type(-1)=CV_64FC3")

            mat_data2 = cv.Mat(mat_data1)
            assert isinstance(mat_data2, cv.Mat)
            assert isinstance(mat_data2, np.ndarray)
            self.assertEqual(mat_data2.wrap_channels, True)  # __array_finalize__ doesn't work
            res2 = cv.utils.dumpInputArray(mat_data2)
            self.assertEqual(res2, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=500 dims(-1)=3 size(-1)=[5 10 10] type(-1)=CV_64FC3")


        def test_mat_wrap_channels_fail(self):
            data = np.random.random([2, 3, 4, 520])

            mat_data0 = cv.Mat(data)
            assert isinstance(mat_data0, cv.Mat)
            assert isinstance(mat_data0, np.ndarray)
            self.assertEqual(mat_data0.wrap_channels, False)
            res0 = cv.utils.dumpInputArray(mat_data0)
            self.assertEqual(res0, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=12480 dims(-1)=4 size(-1)=[2 3 4 520] type(-1)=CV_64FC1")

            with self.assertRaises(cv.error):
                mat_data1 = cv.Mat(data, wrap_channels=True)  # argument unable to wrap channels, too high (520 > CV_CN_MAX=512)
                res1 = cv.utils.dumpInputArray(mat_data1)
                print(mat_data1.__dict__)
                print(res1)


        def test_ufuncs(self):
            data = np.arange(10)
            mat_data = cv.Mat(data)
            mat_data2 = 2 * mat_data
            self.assertEqual(type(mat_data2), cv.Mat)
            np.testing.assert_equal(2 * data, 2 * mat_data)


        def test_comparison(self):
            # Undefined behavior, do NOT use that.
            # Behavior may be changed in the future

            data = np.ones((10, 10, 3))
            mat_wrapped = cv.Mat(data, wrap_channels=True)
            mat_simple = cv.Mat(data)
            np.testing.assert_equal(mat_wrapped, mat_simple)  # ???: wrap_channels is not checked for now
            np.testing.assert_equal(data, mat_simple)
            np.testing.assert_equal(data, mat_wrapped)

            #self.assertEqual(mat_wrapped, mat_simple)  # ???
            #self.assertTrue(mat_wrapped == mat_simple)  # ???
            #self.assertTrue((mat_wrapped == mat_simple).all())


except unittest.SkipTest as e:

    message = str(e)

    class TestSkip(unittest.TestCase):
        def setUp(self):
            self.skipTest('Skip tests: ' + message)

        def test_skip():
            pass

    pass


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
