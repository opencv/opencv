#!/usr/bin/env python

import cv2 as cv

from tests_common import NewOpenCVTests


class UMatDmaBufBindings(NewOpenCVTests):

    def test_dmabuf_api_is_wrapped(self):
        self.assertTrue(hasattr(cv.ocl, 'createUMatFromDmaBuf'))
        self.assertTrue(hasattr(cv.ocl, 'acquireExternalMemory'))
        self.assertTrue(hasattr(cv.ocl, 'releaseExternalMemory'))

        with self.assertRaises(cv.error):
            cv.ocl.createUMatFromDmaBuf(-1, 4096, 64, 64, 64, cv.CV_8UC1)

        ordinary = cv.UMat(16, 16, cv.CV_8UC1)
        with self.assertRaises(cv.error):
            cv.ocl.acquireExternalMemory(ordinary)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
