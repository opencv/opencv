import unittest

import cv2 as cv
import numpy as np


class NumpyByteOrderTest(unittest.TestCase):

    def test_numpy_non_native_byte_order_input(self):
        for dtype in (np.int16, np.uint16, np.int32, np.int64, np.float16, np.float32, np.float64):
            for strided in (False, True):
                native = np.arange(120, dtype=dtype).reshape(10, 12)
                foreign = native.astype(native.dtype.newbyteorder('S'))
                if strided:
                    native = native[:, ::2]
                    foreign = foreign[:, ::2]
                foreign.setflags(write=False)
                before = foreign.tobytes()
                expected = cv.copyTo(native, None)
                actual = cv.copyTo(foreign, None)
                self.assertEqual(actual.dtype, expected.dtype)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(foreign.tobytes(), before)
                self.assertFalse(foreign.flags.writeable)

    def test_numpy_non_native_byte_order_normalize(self):
        for dtype in (np.float32, np.float64):
            native = np.linspace(0, 1, 120, dtype=dtype).reshape(10, 12)
            foreign = native.astype(native.dtype.newbyteorder('S'))
            expected = cv.normalize(native, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
            actual = cv.normalize(foreign, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
            np.testing.assert_array_equal(actual, expected)

    def test_numpy_non_native_byte_order_output_rejected(self):
        for dtype in (np.int16, np.float16, np.float32, np.float64):
            output = np.full((10, 12), 137, dtype=np.dtype(dtype).newbyteorder('S'))
            before = output.tobytes()
            with self.assertRaises(cv.error):
                cv.utils.dumpInputOutputArray(output)
            self.assertEqual(output.tobytes(), before)


if __name__ == '__main__':
    unittest.main()
