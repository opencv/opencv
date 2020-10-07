#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os

from tests_common import NewOpenCVTests

class test_gapi_streaming(NewOpenCVTests):

    def test_image_input(self):
        sz = (1280, 720)
        in_mat = np.random.randint(0, 100, sz).astype(np.uint8)

        # OpenCV
        expected = cv.medianBlur(in_mat, 3)

        # G-API
        g_in = cv.GMat()
        g_out = cv.gapi.medianBlur(g_in, 3)
        c = cv.GComputation(g_in, g_out)
        ccomp = c.compileStreaming(cv.descr_of(cv.gin(in_mat)))
        ccomp.setSource(cv.gin(in_mat))
        ccomp.start()

        _, actual = ccomp.pull()

        # Assert
        self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_video_input(self):
        ksize = 3
        path = self.find_file('cv/video/768x576.avi', [os.environ['OPENCV_TEST_DATA_PATH']])

        # OpenCV
        cap = cv.VideoCapture(path)

        # G-API
        g_in = cv.GMat()
        g_out = cv.gapi.medianBlur(g_in, ksize)
        c = cv.GComputation(g_in, g_out)

        ccomp = c.compileStreaming()
        source = cv.gapi.wip.make_capture_src(path)
        ccomp.setSource(source)
        ccomp.start()

        # Assert
        while cap.isOpened():
            has_expected, expected = cap.read()
            has_actual,   actual   = ccomp.pull()

            self.assertEqual(has_expected, has_actual)

            if not has_actual:
                break

            self.assertEqual(0.0, cv.norm(cv.medianBlur(expected, ksize), actual, cv.NORM_INF))


    def test_video_split3(self):
        path = self.find_file('cv/video/768x576.avi', [os.environ['OPENCV_TEST_DATA_PATH']])

        # OpenCV
        cap = cv.VideoCapture(path)

        # G-API
        g_in = cv.GMat()
        b, g, r = cv.gapi.split3(g_in)
        c = cv.GComputation(cv.GIn(g_in), cv.GOut(b, g, r))

        ccomp = c.compileStreaming()
        source = cv.gapi.wip.make_capture_src(path)
        ccomp.setSource(source)
        ccomp.start()

        # Assert
        while cap.isOpened():
            has_expected, frame = cap.read()
            has_actual,   actual   = ccomp.pull()

            self.assertEqual(has_expected, has_actual)

            if not has_actual:
                break

            expected = cv.split(frame)
            for e, a in zip(expected, actual):
                self.assertEqual(0.0, cv.norm(e, a, cv.NORM_INF))


    def test_video_add(self):
        sz = (576, 768, 3)
        in_mat = np.random.randint(0, 100, sz).astype(np.uint8)

        path = self.find_file('cv/video/768x576.avi', [os.environ['OPENCV_TEST_DATA_PATH']])

        # OpenCV
        cap = cv.VideoCapture(path)

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        out = cv.gapi.add(g_in1, g_in2)
        c = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(out))

        ccomp = c.compileStreaming()
        source = cv.gapi.wip.make_capture_src(path)
        ccomp.setSource(cv.gin(source, in_mat))
        ccomp.start()

        # Assert
        while cap.isOpened():
            has_expected, frame  = cap.read()
            has_actual,   actual = ccomp.pull()

            self.assertEqual(has_expected, has_actual)

            if not has_actual:
                break

            expected = cv.add(frame, in_mat)
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))



if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
