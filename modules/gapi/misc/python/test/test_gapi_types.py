#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os

from tests_common import NewOpenCVTests

class gapi_types_test(NewOpenCVTests):

    def test_garray_type(self):
        types = [cv.gapi.CV_BOOL  , cv.gapi.CV_INT   , cv.gapi.CV_DOUBLE , cv.gapi.CV_FLOAT,
                 cv.gapi.CV_STRING, cv.gapi.CV_POINT , cv.gapi.CV_POINT2F, cv.gapi.CV_SIZE ,
                 cv.gapi.CV_RECT  , cv.gapi.CV_SCALAR, cv.gapi.CV_MAT    , cv.gapi.CV_GMAT]

        for t in types:
            g_array = cv.GArrayT(t)
            self.assertEqual(t, g_array.type())


    def test_gopaque_type(self):
        types = [cv.gapi.CV_BOOL  , cv.gapi.CV_INT   , cv.gapi.CV_DOUBLE , cv.gapi.CV_FLOAT,
                 cv.gapi.CV_STRING, cv.gapi.CV_POINT , cv.gapi.CV_POINT2F, cv.gapi.CV_SIZE ,
                 cv.gapi.CV_RECT]

        for t in types:
            g_opaque = cv.GOpaqueT(t)
            self.assertEqual(t, g_opaque.type())


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
