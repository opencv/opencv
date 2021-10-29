#!/usr/bin/env python
'''
===============================================================================
QR code detect and decode pipeline.
===============================================================================
'''
import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class qrcode_detector_test(NewOpenCVTests):

    def test_detect(self):
        img = cv.imread(os.path.join(self.extraTestDataPath, 'cv/qrcode/link_ocv.jpg'))
        self.assertFalse(img is None)
        detector = cv.QRCodeDetector()
        retval, points = detector.detect(img)
        self.assertTrue(retval)
        self.assertEqual(points.shape, (1, 4, 2))

    def test_detect_and_decode(self):
        img = cv.imread(os.path.join(self.extraTestDataPath, 'cv/qrcode/link_ocv.jpg'))
        self.assertFalse(img is None)
        detector = cv.QRCodeDetector()
        retval, points, straight_qrcode = detector.detectAndDecode(img)
        self.assertEqual(retval, "https://opencv.org/")
        self.assertEqual(points.shape, (1, 4, 2))

    def test_detect_multi(self):
        img = cv.imread(os.path.join(self.extraTestDataPath, 'cv/qrcode/multiple/6_qrcodes.png'))
        self.assertFalse(img is None)
        detector = cv.QRCodeDetector()
        retval, points = detector.detectMulti(img)
        self.assertTrue(retval)
        self.assertEqual(points.shape, (6, 4, 2))

    def test_detect_and_decode_multi(self):
        img = cv.imread(os.path.join(self.extraTestDataPath, 'cv/qrcode/multiple/6_qrcodes.png'))
        self.assertFalse(img is None)
        detector = cv.QRCodeDetector()
        retval, decoded_data, points, straight_qrcode = detector.detectAndDecodeMulti(img)
        self.assertTrue(retval)
        self.assertEqual(len(decoded_data), 6)
        self.assertEqual(decoded_data[0], "TWO STEPS FORWARD")
        self.assertEqual(decoded_data[1], "EXTRA")
        self.assertEqual(decoded_data[2], "SKIP")
        self.assertEqual(decoded_data[3], "STEP FORWARD")
        self.assertEqual(decoded_data[4], "STEP BACK")
        self.assertEqual(decoded_data[5], "QUESTION")
        self.assertEqual(points.shape, (6, 4, 2))
