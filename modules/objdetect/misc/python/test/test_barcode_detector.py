#!/usr/bin/env python
'''
===============================================================================
Barcode detect and decode pipeline.
===============================================================================
'''
import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class barcode_detector_test(NewOpenCVTests):

    def test_detect(self):
        img = cv.imread(os.path.join(self.extraTestDataPath, 'cv/barcode/multiple/4_barcodes.jpg'))
        self.assertFalse(img is None)
        detector = cv.barcode_BarcodeDetector()
        retval, corners = detector.detect(img)
        self.assertTrue(retval)
        self.assertEqual(corners.shape, (4, 4, 2))

    def test_detect_and_decode(self):
        img = cv.imread(os.path.join(self.extraTestDataPath, 'cv/barcode/single/book.jpg'))
        self.assertFalse(img is None)
        detector = cv.barcode_BarcodeDetector()
        retval, decoded_info, decoded_type, corners = detector.detectAndDecodeWithType(img)
        self.assertTrue(retval)
        self.assertTrue(len(decoded_info) > 0)
        self.assertTrue(len(decoded_type) > 0)
        self.assertEqual(decoded_info[0], "9787115279460")
        self.assertEqual(decoded_type[0], "EAN_13")
        self.assertEqual(corners.shape, (1, 4, 2))
