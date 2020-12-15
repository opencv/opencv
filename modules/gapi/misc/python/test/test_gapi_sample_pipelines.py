#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os

from tests_common import NewOpenCVTests


# Plaidml is an optional backend
pkgs = [
         ('ocl'    , cv.gapi.core.ocl.kernels()),
         ('cpu'    , cv.gapi.core.cpu.kernels()),
         ('fluid'  , cv.gapi.core.fluid.kernels())
         # ('plaidml', cv.gapi.core.plaidml.kernels())
     ]


class gapi_sample_pipelines(NewOpenCVTests):

    # NB: This test check multiple outputs for operation
    def test_mean_over_r(self):
        img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
        in_mat = cv.imread(img_path)

        # # OpenCV
        _, _, r_ch = cv.split(in_mat)
        expected = cv.mean(r_ch)

        # G-API
        g_in = cv.GMat()
        b, g, r = cv.gapi.split3(g_in)
        g_out = cv.gapi.mean(r)
        comp = cv.GComputation(g_in, g_out)

        for pkg_name, pkg in pkgs:
            actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF),
                             'Failed on ' + pkg_name + ' backend')


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
