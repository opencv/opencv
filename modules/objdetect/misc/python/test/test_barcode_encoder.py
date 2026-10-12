#!/usr/bin/env python
'''
===============================================================================
Barcode encoding tests for EAN-13, EAN-8, and UPC-A.
===============================================================================
'''
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class barcode_encoder_test(NewOpenCVTests):

    def test_params_default(self):
        p = cv.barcode.BarcodeEncoder.Params()
        self.assertEqual(p.type, cv.barcode.BarcodeEncoder_EAN_13)
        self.assertEqual(p.checksum_policy, cv.barcode.BarcodeEncoder_APPEND_CHECKSUM)
        self.assertEqual(p.module_width, 2)
        self.assertEqual(p.bar_height, 50)

    def test_encode_ean13(self):
        p = cv.barcode.BarcodeEncoder.Params()
        p.type = cv.barcode.BarcodeEncoder_EAN_13
        p.checksum_policy = cv.barcode.BarcodeEncoder_APPEND_CHECKSUM
        p.module_width = 3
        p.bar_height = 80
        encoder = cv.barcode.BarcodeEncoder.create(p)

        # 12 digits -> check digit 0 appended -> 9787115279460
        # Total modules: 11 (left quiet) + 95 (symbol) + 7 (right quiet) = 113 modules
        img = encoder.encode('978711527946')
        self.assertEqual(img.shape, (80, 113 * 3))
        self.assertEqual(img.dtype, np.uint8)

        # Roundtrip decode
        detector = cv.barcode_BarcodeDetector()
        w = img.shape[1]
        h = img.shape[0]
        corners = np.array([[[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]]], dtype=np.float32)
        retval, decoded_info, decoded_type = detector.decodeWithType(img, corners)
        self.assertTrue(retval)
        self.assertEqual(decoded_info[0], '9787115279460')
        self.assertEqual(decoded_type[0], 'EAN_13')

    def test_encode_ean8(self):
        p = cv.barcode.BarcodeEncoder.Params()
        p.type = cv.barcode.BarcodeEncoder_EAN_8
        p.checksum_policy = cv.barcode.BarcodeEncoder_APPEND_CHECKSUM
        p.module_width = 2
        p.bar_height = 60
        encoder = cv.barcode.BarcodeEncoder.create(p)

        # 7 digits -> check digit 5 appended -> 40170725
        # Total modules: 7 (left quiet) + 67 (symbol) + 7 (right quiet) = 81 modules
        img = encoder.encode('4017072')
        self.assertEqual(img.shape, (60, 81 * 2))

        # Roundtrip decode
        detector = cv.barcode_BarcodeDetector()
        w = img.shape[1]
        h = img.shape[0]
        corners = np.array([[[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]]], dtype=np.float32)
        retval, decoded_info, decoded_type = detector.decodeWithType(img, corners)
        self.assertTrue(retval)
        self.assertEqual(decoded_info[0], '40170725')
        self.assertEqual(decoded_type[0], 'EAN_8')

    def test_encode_upca(self):
        p = cv.barcode.BarcodeEncoder.Params()
        p.type = cv.barcode.BarcodeEncoder_UPC_A
        p.checksum_policy = cv.barcode.BarcodeEncoder_APPEND_CHECKSUM
        p.module_width = 2
        p.bar_height = 60
        encoder = cv.barcode.BarcodeEncoder.create(p)

        # 11 digits -> check digit 5 appended -> 012345678905
        # Total modules: 9 (left quiet) + 95 (symbol) + 9 (right quiet) = 113 modules
        img = encoder.encode('01234567890')
        self.assertEqual(img.shape, (60, 113 * 2))

        # Roundtrip decode
        detector = cv.barcode_BarcodeDetector()
        w = img.shape[1]
        h = img.shape[0]
        corners = np.array([[[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]]], dtype=np.float32)
        retval, decoded_info, decoded_type = detector.decodeWithType(img, corners)
        self.assertTrue(retval)
        self.assertEqual(decoded_info[0], '012345678905')
        self.assertEqual(decoded_type[0], 'UPC_A')

    def test_checksum_validation(self):
        p = cv.barcode.BarcodeEncoder.Params()
        p.type = cv.barcode.BarcodeEncoder_EAN_13
        p.checksum_policy = cv.barcode.BarcodeEncoder_VALIDATE_CHECKSUM
        encoder = cv.barcode.BarcodeEncoder.create(p)

        # Valid checksum should succeed
        img = encoder.encode('9787115279460')
        self.assertFalse(img is None)

        # Invalid checksum should raise exception
        with self.assertRaises(cv.error):
            encoder.encode('9787115279461')

    def test_invalid_input(self):
        p = cv.barcode.BarcodeEncoder.Params()
        p.type = cv.barcode.BarcodeEncoder_EAN_13
        encoder = cv.barcode.BarcodeEncoder.create(p)

        # Non-numeric characters
        with self.assertRaises(cv.error):
            encoder.encode('97871152794A')

        # Invalid length
        with self.assertRaises(cv.error):
            encoder.encode('12345')
