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
