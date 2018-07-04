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


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
