#!/usr/bin/env python
'''
===============================================================================
QR code detect and decode pipeline.
===============================================================================
'''

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class qrcode_detector_test(NewOpenCVTests):
    def test_detect_and_decode(self):
        img = cv.imread(self.extraTestDataPath + '/cv/qrcode/link_ocv.jpg')
        detector = cv.QRCodeDetector()
        retval, points, straight_qrcode = detector.detectAndDecode(img)
        self.assertEqual(retval, "https://opencv.org/");
    def test_multiple_detect_and_decode(self):
        img = cv.imread(self.extraTestDataPath + '/cv/qrcode/multiple/6_qrcodes.png')
        detector = cv.QRCodeDetector()
        retval, points, straight_qrcode = detector.multipleDetectAndDecode(img)
        self.assertEqual(retval[0], "TWO STEPS FORWARD");
        self.assertEqual(retval[1], "QUESTION");
        self.assertEqual(retval[2], "SKIP");
        self.assertEqual(retval[3], "EXTRA");
        self.assertEqual(retval[4], "STEP FORWARD");
        self.assertEqual(retval[5], "STEP BACK");
