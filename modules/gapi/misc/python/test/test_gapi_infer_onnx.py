#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os
import sys
import unittest

from tests_common import NewOpenCVTests


try:

    if sys.version_info[:2] < (3, 0):
        raise unittest.SkipTest('Python 2.x is not supported')

    CLASSIFICATION_MODEL_PATH = "vision/classification/squeezenet/model/squeezenet1.0-9.onnx"

    class test_gapi_infer(NewOpenCVTests):
        def find_dnn_file(self, filename):
            return self.find_file(filename, [os.environ.get('OPENCV_GAPI_ONNX_MODEL_PATH')], False)

        def test_onnx_classification(self):
            model_path = self.find_dnn_file(CLASSIFICATION_MODEL_PATH)
            if model_path is None:
                raise unittest.SkipTest("Missing DNN test file")

            in_mat = cv.imread(
                self.find_file("cv/dpm/cat.png",
                [os.environ.get('OPENCV_TEST_DATA_PATH')]))

            g_in = cv.GMat()
            g_infer_inputs = cv.GInferInputs()
            g_infer_inputs.setInput("data_0", g_in)
            g_infer_out = cv.gapi.infer("squeeze-net", g_infer_inputs)
            g_out = g_infer_out.at("softmaxout_1")

            comp = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))

            net = cv.gapi.onnx.params("squeeze-net", model_path)
            net.cfgNormalize("data_0", False)
            try:
                out_gapi = comp.apply(cv.gin(in_mat), cv.gapi.compile_args(cv.gapi.networks(net)))
            except cv.error as err:
                if err.args[0] == "G-API has been compiled without ONNX support":
                    raise unittest.SkipTest("G-API has been compiled without ONNX support")
                else:
                    raise

            self.assertEqual((1, 1000, 1, 1), out_gapi.shape)


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
