#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os
import sys
import unittest
import time

from tests_common import NewOpenCVTests


try:
    if sys.version_info[:2] < (3, 0):
        raise unittest.SkipTest('Python 2.x is not supported')


    @cv.gapi.op('custom.delay', in_types=[cv.GMat], out_types=[cv.GMat])
    class GDelay:
        """Delay for 10 ms."""

        @staticmethod
        def outMeta(desc):
            return desc


    @cv.gapi.kernel(GDelay)
    class GDelayImpl:
        """Implementation for GDelay operation."""

        @staticmethod
        def run(img):
            time.sleep(0.01)
            return img


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
            ccomp = c.compileStreaming(cv.gapi.descr_of(in_mat))
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
            ccomp.setSource(cv.gin(source))
            ccomp.start()

            # Assert
            max_num_frames  = 10
            proc_num_frames = 0
            while cap.isOpened():
                has_expected, expected = cap.read()
                has_actual,   actual   = ccomp.pull()

                self.assertEqual(has_expected, has_actual)

                if not has_actual:
                    break

                self.assertEqual(0.0, cv.norm(cv.medianBlur(expected, ksize), actual, cv.NORM_INF))

                proc_num_frames += 1
                if proc_num_frames == max_num_frames:
                    break


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
            ccomp.setSource(cv.gin(source))
            ccomp.start()

            # Assert
            max_num_frames  = 10
            proc_num_frames = 0
            while cap.isOpened():
                has_expected, frame = cap.read()
                has_actual,   actual   = ccomp.pull()

                self.assertEqual(has_expected, has_actual)

                if not has_actual:
                    break

                expected = cv.split(frame)
                for e, a in zip(expected, actual):
                    self.assertEqual(0.0, cv.norm(e, a, cv.NORM_INF))

                proc_num_frames += 1
                if proc_num_frames == max_num_frames:
                    break


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
            max_num_frames  = 10
            proc_num_frames = 0
            while cap.isOpened():
                has_expected, frame  = cap.read()
                has_actual,   actual = ccomp.pull()

                self.assertEqual(has_expected, has_actual)

                if not has_actual:
                    break

                expected = cv.add(frame, in_mat)
                self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))

                proc_num_frames += 1
                if proc_num_frames == max_num_frames:
                    break


        def test_video_good_features_to_track(self):
            path = self.find_file('cv/video/768x576.avi', [os.environ['OPENCV_TEST_DATA_PATH']])

            # NB: goodFeaturesToTrack configuration
            max_corners         = 50
            quality_lvl         = 0.01
            min_distance        = 10
            block_sz            = 3
            use_harris_detector = True
            k                   = 0.04
            mask                = None

            # OpenCV
            cap = cv.VideoCapture(path)

            # G-API
            g_in = cv.GMat()
            g_gray = cv.gapi.RGB2Gray(g_in)
            g_out = cv.gapi.goodFeaturesToTrack(g_gray, max_corners, quality_lvl,
                                                min_distance, mask, block_sz, use_harris_detector, k)

            c = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))

            ccomp = c.compileStreaming()
            source = cv.gapi.wip.make_capture_src(path)
            ccomp.setSource(cv.gin(source))
            ccomp.start()

            # Assert
            max_num_frames  = 10
            proc_num_frames = 0
            while cap.isOpened():
                has_expected, frame  = cap.read()
                has_actual,   actual = ccomp.pull()

                self.assertEqual(has_expected, has_actual)

                if not has_actual:
                    break

                # OpenCV
                frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                expected = cv.goodFeaturesToTrack(frame, max_corners, quality_lvl,
                                                  min_distance, mask=mask,
                                                  blockSize=block_sz, useHarrisDetector=use_harris_detector, k=k)
                for e, a in zip(expected, actual):
                    # NB: OpenCV & G-API have different output shapes:
                    # OpenCV - (num_points, 1, 2)
                    # G-API  - (num_points, 2)
                    self.assertEqual(0.0, cv.norm(e.flatten(),
                                                  np.array(a, np.float32).flatten(),
                                                  cv.NORM_INF))

                proc_num_frames += 1
                if proc_num_frames == max_num_frames:
                    break


        def test_gapi_streaming_meta(self):
            ksize = 3
            path = self.find_file('cv/video/768x576.avi', [os.environ['OPENCV_TEST_DATA_PATH']])

            # G-API
            g_in = cv.GMat()
            g_ts = cv.gapi.streaming.timestamp(g_in)
            g_seqno = cv.gapi.streaming.seqNo(g_in)
            g_seqid = cv.gapi.streaming.seq_id(g_in)

            c = cv.GComputation(cv.GIn(g_in), cv.GOut(g_ts, g_seqno, g_seqid))

            ccomp = c.compileStreaming()
            source = cv.gapi.wip.make_capture_src(path)
            ccomp.setSource(cv.gin(source))
            ccomp.start()

            # Assert
            max_num_frames  = 10
            curr_frame_number = 0
            while True:
                has_frame, (ts, seqno, seqid) = ccomp.pull()

                if not has_frame:
                    break

                self.assertEqual(curr_frame_number, seqno)
                self.assertEqual(curr_frame_number, seqid)

                curr_frame_number += 1
                if curr_frame_number == max_num_frames:
                    break


        def test_desync(self):
            path = self.find_file('cv/video/768x576.avi', [os.environ['OPENCV_TEST_DATA_PATH']])

            # G-API
            g_in = cv.GMat()
            g_out1 = cv.gapi.copy(g_in)
            des = cv.gapi.streaming.desync(g_in)
            g_out2 = GDelay.on(des)

            c = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out1, g_out2))

            kernels = cv.gapi.kernels(GDelayImpl)
            ccomp = c.compileStreaming(args=cv.gapi.compile_args(kernels))
            source = cv.gapi.wip.make_capture_src(path)
            ccomp.setSource(cv.gin(source))
            ccomp.start()

            # Assert
            max_num_frames  = 10
            proc_num_frames = 0

            out_counter = 0
            desync_out_counter = 0
            none_counter = 0
            while True:
                has_frame, (out1, out2) = ccomp.pull()
                if not has_frame:
                    break

                if not out1 is None:
                    out_counter += 1
                if not out2 is None:
                    desync_out_counter += 1
                else:
                    none_counter += 1

                proc_num_frames += 1
                if proc_num_frames == max_num_frames:
                    ccomp.stop()
                    break

            self.assertLess(0, proc_num_frames)
            self.assertLess(desync_out_counter, out_counter)
            self.assertLess(0, none_counter)


        def test_compile_streaming_empty(self):
            g_in = cv.GMat()
            comp = cv.GComputation(g_in, cv.gapi.medianBlur(g_in, 3))
            comp.compileStreaming()


        def test_compile_streaming_args(self):
            g_in = cv.GMat()
            comp = cv.GComputation(g_in, cv.gapi.medianBlur(g_in, 3))
            comp.compileStreaming(cv.gapi.compile_args(cv.gapi.streaming.queue_capacity(1)))


        def test_compile_streaming_descr_of(self):
            g_in = cv.GMat()
            comp = cv.GComputation(g_in, cv.gapi.medianBlur(g_in, 3))
            img = np.zeros((3,300,300), dtype=np.float32)
            comp.compileStreaming(cv.gapi.descr_of(img))


        def test_compile_streaming_descr_of_and_args(self):
            g_in = cv.GMat()
            comp = cv.GComputation(g_in, cv.gapi.medianBlur(g_in, 3))
            img = np.zeros((3,300,300), dtype=np.float32)
            comp.compileStreaming(cv.gapi.descr_of(img),
                    cv.gapi.compile_args(cv.gapi.streaming.queue_capacity(1)))


        def test_compile_streaming_meta(self):
            g_in = cv.GMat()
            comp = cv.GComputation(g_in, cv.gapi.medianBlur(g_in, 3))
            img = np.zeros((3,300,300), dtype=np.float32)
            comp.compileStreaming([cv.GMatDesc(cv.CV_8U, 3, (300, 300))])


        def test_compile_streaming_meta_and_args(self):
            g_in = cv.GMat()
            comp = cv.GComputation(g_in, cv.gapi.medianBlur(g_in, 3))
            img = np.zeros((3,300,300), dtype=np.float32)
            comp.compileStreaming([cv.GMatDesc(cv.CV_8U, 3, (300, 300))],
                    cv.gapi.compile_args(cv.gapi.streaming.queue_capacity(1)))



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
