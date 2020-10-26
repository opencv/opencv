#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os

from tests_common import NewOpenCVTests


class test_gapi_infer(NewOpenCVTests):

    def test_getAvailableTargets(self):
        targets = cv.dnn.getAvailableTargets(cv.dnn.DNN_BACKEND_OPENCV)
        self.assertTrue(cv.dnn.DNN_TARGET_CPU in targets)


    def test_age_gender_infer(self):

        # NB: Check IE
        if not cv.dnn.DNN_TARGET_CPU in cv.dnn.getAvailableTargets(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE):
            return

        root_path    = '/omz_intel_models/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013'
        model_path   = self.find_file(root_path + '.xml',   [os.environ.get('OPENCV_DNN_TEST_DATA_PATH')])
        weights_path = self.find_file(root_path + '.bin',   [os.environ.get('OPENCV_DNN_TEST_DATA_PATH')])
        img_path     = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
        device_id    = 'CPU'
        img          = cv.resize(cv.imread(img_path), (62,62))

        # OpenCV DNN
        net = cv.dnn.readNetFromModelOptimizer(model_path, weights_path)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        blob = cv.dnn.blobFromImage(img)

        net.setInput(blob)
        dnn_age, dnn_gender = net.forward(net.getUnconnectedOutLayersNames())

        # OpenCV G-API
        g_in   = cv.GMat()
        inputs = cv.GInferInputs()
        inputs.setInput('data', g_in)

        outputs  = cv.gapi.infer("net", inputs)
        age_g    = outputs.at("age_conv3")
        gender_g = outputs.at("prob")

        comp = cv.GComputation(cv.GIn(g_in), cv.GOut(age_g, gender_g))
        pp = cv.gapi.ie.params("net", model_path, weights_path, device_id)

        nets = cv.gapi.networks(pp)
        args = cv.compile_args(nets)
        gapi_age, gapi_gender = comp.apply(cv.gin(img), args=cv.compile_args(cv.gapi.networks(pp)))

        # Check
        self.assertEqual(0.0, cv.norm(dnn_gender, gapi_gender, cv.NORM_INF))
        self.assertEqual(0.0, cv.norm(dnn_age, gapi_age, cv.NORM_INF))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
