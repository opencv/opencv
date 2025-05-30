# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
===============================================================================
QR code detect and decode pipeline.
===============================================================================
'''
import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests, unittest

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
        self.assertTrue("TWO STEPS FORWARD" in decoded_data)
        self.assertTrue("EXTRA" in decoded_data)
        self.assertTrue("SKIP" in decoded_data)
        self.assertTrue("STEP FORWARD" in decoded_data)
        self.assertTrue("STEP BACK" in decoded_data)
        self.assertTrue("QUESTION" in decoded_data)
        self.assertEqual(points.shape, (6, 4, 2))

    def test_decode_non_ascii(self):
        import sys
        if sys.version_info[0] < 3:
            raise unittest.SkipTest('Python 2.x is not supported')

        img = cv.imread(os.path.join(self.extraTestDataPath, 'cv/qrcode/umlaut.png'))
        self.assertFalse(img is None)
        detector = cv.QRCodeDetector()
        decoded_data, _, _ = detector.detectAndDecode(img)
        self.assertTrue(isinstance(decoded_data, str))
        self.assertTrue("Müllheimstrasse" in decoded_data)

    def test_kanji(self):
        inp = b"\x82\xb1\x82\xf1\x82\xc9\x82\xbf\x82\xcd\x90\xa2\x8a\x45"

        params = cv.QRCodeEncoder_Params()
        params.mode = cv.QRCodeEncoder_MODE_KANJI
        encoder = cv.QRCodeEncoder_create(params)
        qrcode = encoder.encode(inp)
        qrcode = cv.resize(qrcode, (0, 0), fx=2, fy=2, interpolation=cv.INTER_NEAREST)

        detector = cv.QRCodeDetector()
        data, _, _ = detector.detectAndDecodeBytes(qrcode)
        self.assertEqual(data, inp)
        self.assertEqual(detector.getEncoding(), cv.QRCodeEncoder_ECI_SHIFT_JIS)
        self.assertEqual(data.decode("shift-jis"), "こんにちは世界")
