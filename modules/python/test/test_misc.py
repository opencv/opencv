#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class Bindings(NewOpenCVTests):

    def test_inheritance(self):
        bm = cv.StereoBM_create()
        bm.getPreFilterCap() # from StereoBM
        bm.getBlockSize() # from SteroMatcher

        boost = cv.ml.Boost_create()
        boost.getBoostType() # from ml::Boost
        boost.getMaxDepth() # from ml::DTrees
        boost.isClassifier() # from ml::StatModel


    def test_redirectError(self):
        try:
            cv.imshow("", None) # This causes an assert
            self.assertEqual("Dead code", 0)
        except cv.error as _e:
            pass

        handler_called = [False]
        def test_error_handler(status, func_name, err_msg, file_name, line):
            handler_called[0] = True

        cv.redirectError(test_error_handler)
        try:
            cv.imshow("", None) # This causes an assert
            self.assertEqual("Dead code", 0)
        except cv.error as _e:
            self.assertEqual(handler_called[0], True)
            pass

        cv.redirectError(None)
        try:
            cv.imshow("", None) # This causes an assert
            self.assertEqual("Dead code", 0)
        except cv.error as _e:
            pass


class Arguments(NewOpenCVTests):

    def test_InputArray(self):
        res1 = cv.utils.dumpInputArray(None)
        #self.assertEqual(res1, "InputArray: noArray()")  # not supported
        self.assertEqual(res1, "InputArray: empty()=true kind=0x00010000 flags=0x01010000 total(-1)=0 dims(-1)=0 size(-1)=0x0 type(-1)=CV_8UC1")
        res2_1 = cv.utils.dumpInputArray((1, 2))
        self.assertEqual(res2_1, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=2 dims(-1)=2 size(-1)=1x2 type(-1)=CV_64FC1")
        res2_2 = cv.utils.dumpInputArray(1.5)  # Scalar(1.5, 1.5, 1.5, 1.5)
        self.assertEqual(res2_2, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=4 dims(-1)=2 size(-1)=1x4 type(-1)=CV_64FC1")
        a = np.array([[1,2],[3,4],[5,6]])
        res3 = cv.utils.dumpInputArray(a)  # 32SC1
        self.assertEqual(res3, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=6 dims(-1)=2 size(-1)=2x3 type(-1)=CV_32SC1")
        a = np.array([[[1,2],[3,4],[5,6]]], dtype='f')
        res4 = cv.utils.dumpInputArray(a)  # 32FC2
        self.assertEqual(res4, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=3 dims(-1)=2 size(-1)=3x1 type(-1)=CV_32FC2")
        a = np.array([[[1,2]],[[3,4]],[[5,6]]], dtype=float)
        res5 = cv.utils.dumpInputArray(a)  # 64FC2
        self.assertEqual(res5, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=3 dims(-1)=2 size(-1)=1x3 type(-1)=CV_64FC2")


    def test_InputArrayOfArrays(self):
        res1 = cv.utils.dumpInputArrayOfArrays(None)
        #self.assertEqual(res1, "InputArray: noArray()")  # not supported
        self.assertEqual(res1, "InputArrayOfArrays: empty()=true kind=0x00050000 flags=0x01050000 total(-1)=0 dims(-1)=1 size(-1)=0x0")
        res2_1 = cv.utils.dumpInputArrayOfArrays((1, 2))  # { Scalar:all(1), Scalar::all(2) }
        self.assertEqual(res2_1, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=2 dims(-1)=1 size(-1)=2x1 type(0)=CV_64FC1 dims(0)=2 size(0)=1x4 type(0)=CV_64FC1")
        res2_2 = cv.utils.dumpInputArrayOfArrays([1.5])
        self.assertEqual(res2_2, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=1 dims(-1)=1 size(-1)=1x1 type(0)=CV_64FC1 dims(0)=2 size(0)=1x4 type(0)=CV_64FC1")
        a = np.array([[1,2],[3,4],[5,6]])
        b = np.array([[1,2,3],[4,5,6],[7,8,9]])
        res3 = cv.utils.dumpInputArrayOfArrays([a, b])
        self.assertEqual(res3, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=2 dims(-1)=1 size(-1)=2x1 type(0)=CV_32SC1 dims(0)=2 size(0)=2x3 type(0)=CV_32SC1")
        c = np.array([[[1,2],[3,4],[5,6]]], dtype='f')
        res4 = cv.utils.dumpInputArrayOfArrays([c, a, b])
        self.assertEqual(res4, "InputArrayOfArrays: empty()=false kind=0x00050000 flags=0x01050000 total(-1)=3 dims(-1)=1 size(-1)=3x1 type(0)=CV_32FC2 dims(0)=2 size(0)=3x1 type(0)=CV_32FC2")



if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
