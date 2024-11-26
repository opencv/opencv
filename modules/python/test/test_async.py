#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class AsyncTest(NewOpenCVTests):

    def test_async_simple(self):
        m = np.array([[1,2],[3,4],[5,6]], dtype=np.intc)
        async_result = cv.utils.testAsyncArray(m)
        self.assertTrue(async_result.valid())
        ret, result = async_result.get(timeoutNs=10**6)  # 1ms
        self.assertTrue(ret)
        self.assertFalse(async_result.valid())
        self.assertEqual(cv.norm(m, result, cv.NORM_INF), 0)


    def test_async_exception(self):
        async_result = cv.utils.testAsyncException()
        self.assertTrue(async_result.valid())
        try:
            _ret, _result = async_result.get(timeoutNs=10**6)  # 1ms
            self.fail("Exception expected")
        except cv.error as e:
            self.assertEqual(cv.Error.StsOk, e.code)



if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
