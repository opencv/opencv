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

    def test_module_width_overflow(self):
        p = cv.barcode.BarcodeEncoder.Params()
        p.type = cv.barcode.BarcodeEncoder_EAN_13
        p.module_width = (2147483647 // 113) + 1
        encoder = cv.barcode.BarcodeEncoder.create(p)
        with self.assertRaises(cv.error):
            encoder.encode('978711527946')

    def test_reference_bitstrings(self):
        def to_bits(img_row):
            return ''.join('1' if px == 0 else '0' for px in img_row)

        p = cv.barcode.BarcodeEncoder.Params()
        p.module_width = 1
        p.bar_height = 20

        # EAN-13
        p.type = cv.barcode.BarcodeEncoder_EAN_13
        p.checksum_policy = cv.barcode.BarcodeEncoder_VALIDATE_CHECKSUM
        enc_ean13 = cv.barcode.BarcodeEncoder.create(p)
        img_ean13 = enc_ean13.encode('6291041500213')
        expected_ean13 = (
            '00000000000'
            '101'
            '001001100101110110011010011101000110011001'
            '01010'
            '100111011100101110010110110011001101000010'
            '101'
            '0000000'
        )
        self.assertEqual(to_bits(img_ean13[0]), expected_ean13)

        # EAN-8
        p.type = cv.barcode.BarcodeEncoder_EAN_8
        enc_ean8 = cv.barcode.BarcodeEncoder.create(p)
        img_ean8 = enc_ean8.encode('40170725')
        expected_ean8 = (
            '0000000'
            '101'
            '0100011000110100110010111011'
            '01010'
            '1110010100010011011001001110'
            '101'
            '0000000'
        )
        self.assertEqual(to_bits(img_ean8[0]), expected_ean8)

        # UPC-A
        p.type = cv.barcode.BarcodeEncoder_UPC_A
        enc_upca = cv.barcode.BarcodeEncoder.create(p)
        img_upca = enc_upca.encode('012345678905')
        expected_upca = (
            '000000000'
            '101'
            '000110100110010010011011110101000110110001'
            '01010'
            '101000010001001001000111010011100101001110'
            '101'
            '000000000'
        )
        self.assertEqual(to_bits(img_upca[0]), expected_upca)
