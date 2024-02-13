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


    openvino_is_available = True
    try:
        from openvino.runtime import Core, Type, Layout, PartialShape
        from openvino.preprocess import ResizeAlgorithm, PrePostProcessor
    except ImportError:
        openvino_is_available = False


    def skip_if_openvino_not_available():
        if not openvino_is_available:
            raise unittest.SkipTest("OpenVINO isn't available from python.")


    class AgeGenderOV:
        def __init__(self, model_path, bin_path, device):
            self.device = device
            self.core = Core()
            self.model = self.core.read_model(model_path, bin_path)


        def reshape(self, new_shape):
            self.model.reshape(new_shape)


        def cfgPrePostProcessing(self, pp_callback):
            ppp = PrePostProcessor(self.model)
            pp_callback(ppp)
            self.model = ppp.build()


        def apply(self, in_data):
           compiled_model = self.core.compile_model(self.model, self.device)
           infer_request = compiled_model.create_infer_request()
           results = infer_request.infer(in_data)
           ov_age = results['age_conv3'].squeeze()
           ov_gender = results['prob'].squeeze()
           return ov_age, ov_gender


    class AgeGenderGAPI:
        tag = 'age-gender-net'

        def __init__(self, model_path, bin_path, device):
            g_in   = cv.GMat()
            inputs = cv.GInferInputs()
            inputs.setInput('data', g_in)
            # TODO: It'd be nice to pass dict instead.
            # E.g cv.gapi.infer("net", {'data': g_in})
            outputs = cv.gapi.infer(AgeGenderGAPI.tag, inputs)
            age_g = outputs.at("age_conv3")
            gender_g = outputs.at("prob")

            self.comp = cv.GComputation(cv.GIn(g_in), cv.GOut(age_g, gender_g))
            self.pp = cv.gapi.ov.params(AgeGenderGAPI.tag, \
                                        model_path, bin_path, device)


        def apply(self, in_data):
           compile_args = cv.gapi.compile_args(cv.gapi.networks(self.pp))
           gapi_age, gapi_gender = self.comp.apply(cv.gin(in_data), compile_args)
           gapi_gender = gapi_gender.squeeze()
           gapi_age = gapi_age.squeeze()
           return gapi_age, gapi_gender


    class test_gapi_infer_ov(NewOpenCVTests):

        def test_age_gender_infer_image(self):
            skip_if_openvino_not_available()

            root_path  = '/omz_intel_models/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013'
            model_path = self.find_file(root_path + '.xml',   [os.environ.get('OPENCV_DNN_TEST_DATA_PATH')], required=False)
            bin_path   = self.find_file(root_path + '.bin',   [os.environ.get('OPENCV_DNN_TEST_DATA_PATH')], required=False)
            device_id  = 'CPU'

            img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
            img = cv.imread(img_path)

            # OpenVINO
            def preproc(ppp):
                ppp.input().model().set_layout(Layout("NCHW"))
                ppp.input().tensor().set_element_type(Type.u8)                            \
                                    .set_spatial_static_shape(img.shape[0], img.shape[1]) \
                                    .set_layout(Layout("NHWC"))
                ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)


            ref = AgeGenderOV(model_path, bin_path, device_id)
            ref.cfgPrePostProcessing(preproc)
            ov_age, ov_gender = ref.apply(np.expand_dims(img, 0))

            # OpenCV G-API (No preproc required)
            comp = AgeGenderGAPI(model_path, bin_path, device_id)
            gapi_age, gapi_gender = comp.apply(img)

            # Check
            self.assertEqual(0.0, cv.norm(ov_gender, gapi_gender, cv.NORM_INF))
            self.assertEqual(0.0, cv.norm(ov_age, gapi_age, cv.NORM_INF))


        def test_age_gender_infer_tensor(self):
            skip_if_openvino_not_available()

            root_path  = '/omz_intel_models/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013'
            model_path = self.find_file(root_path + '.xml',   [os.environ.get('OPENCV_DNN_TEST_DATA_PATH')], required=False)
            bin_path   = self.find_file(root_path + '.bin',   [os.environ.get('OPENCV_DNN_TEST_DATA_PATH')], required=False)
            device_id  = 'CPU'

            img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
            img = cv.imread(img_path)

            # Prepare data manually
            tensor = cv.resize(img, (62, 62)).astype(np.float32)
            tensor = np.transpose(tensor, (2, 0, 1))
            tensor = np.expand_dims(tensor, 0)

            # OpenVINO (No preproce required)
            ref = AgeGenderOV(model_path, bin_path, device_id)
            ov_age, ov_gender = ref.apply(tensor)

            # OpenCV G-API (No preproc required)
            comp = AgeGenderGAPI(model_path, bin_path, device_id)
            gapi_age, gapi_gender = comp.apply(tensor)

            # Check
            self.assertEqual(0.0, cv.norm(ov_gender, gapi_gender, cv.NORM_INF))
            self.assertEqual(0.0, cv.norm(ov_age, gapi_age, cv.NORM_INF))


        def test_age_gender_infer_batch(self):
            skip_if_openvino_not_available()

            root_path  = '/omz_intel_models/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013'
            model_path = self.find_file(root_path + '.xml',   [os.environ.get('OPENCV_DNN_TEST_DATA_PATH')], required=False)
            bin_path   = self.find_file(root_path + '.bin',   [os.environ.get('OPENCV_DNN_TEST_DATA_PATH')], required=False)
            device_id  = 'CPU'

            img_path1 = self.find_file('cv/face/david1.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
            img_path2 = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
            img1 = cv.imread(img_path1)
            img2 = cv.imread(img_path2)
            # img1 and img2 have the same size
            batch_img = np.array([img1, img2])

            # OpenVINO
            def preproc(ppp):
                ppp.input().model().set_layout(Layout("NCHW"))
                ppp.input().tensor().set_element_type(Type.u8)                              \
                                    .set_spatial_static_shape(img1.shape[0], img2.shape[1]) \
                                    .set_layout(Layout("NHWC"))
                ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)


            ref = AgeGenderOV(model_path, bin_path, device_id)
            ref.reshape(PartialShape([2, 3, 62, 62]))
            ref.cfgPrePostProcessing(preproc)
            ov_age, ov_gender = ref.apply(batch_img)

            # OpenCV G-API
            comp = AgeGenderGAPI(model_path, bin_path, device_id)
            comp.pp.cfgReshape([2, 3, 62, 62])   \
                   .cfgInputModelLayout("NCHW")  \
                   .cfgInputTensorLayout("NHWC") \
                   .cfgResize(cv.INTER_LINEAR)
            gapi_age, gapi_gender = comp.apply(batch_img)

            # Check
            self.assertEqual(0.0, cv.norm(ov_gender, gapi_gender, cv.NORM_INF))
            self.assertEqual(0.0, cv.norm(ov_age, gapi_age, cv.NORM_INF))


        def test_age_gender_infer_planar(self):
            skip_if_openvino_not_available()

            root_path  = '/omz_intel_models/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013'
            model_path = self.find_file(root_path + '.xml',   [os.environ.get('OPENCV_DNN_TEST_DATA_PATH')], required=False)
            bin_path   = self.find_file(root_path + '.bin',   [os.environ.get('OPENCV_DNN_TEST_DATA_PATH')], required=False)
            device_id  = 'CPU'

            img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
            img = cv.imread(img_path)
            planar_img = np.transpose(img, (2, 0, 1))
            planar_img = np.expand_dims(planar_img, 0)

            # OpenVINO
            def preproc(ppp):
                ppp.input().tensor().set_element_type(Type.u8)                            \
                                    .set_spatial_static_shape(img.shape[0], img.shape[1])
                ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)


            ref = AgeGenderOV(model_path, bin_path, device_id)
            ref.cfgPrePostProcessing(preproc)
            ov_age, ov_gender = ref.apply(planar_img)

            # OpenCV G-API
            comp = AgeGenderGAPI(model_path, bin_path, device_id)
            comp.pp.cfgResize(cv.INTER_LINEAR)
            gapi_age, gapi_gender = comp.apply(planar_img)

            # Check
            self.assertEqual(0.0, cv.norm(ov_gender, gapi_gender, cv.NORM_INF))
            self.assertEqual(0.0, cv.norm(ov_age, gapi_age, cv.NORM_INF))


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
