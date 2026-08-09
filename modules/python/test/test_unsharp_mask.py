import cv2 as cv
import numpy as np
from tests_common import NewOpenCVTests

class UnsharpMaskTest(NewOpenCVTests):
    def test_unsharp_mask_basic_uint8(self):
        src = np.full((100, 100), 50, dtype=np.uint8)
        src[:, 50:] = 200

        dst = cv.unsharpMask(src, sigma=1.0, amount=1.0, threshold=0.0)
        self.assertEqual(dst.shape, src.shape)
        self.assertEqual(dst.dtype, np.uint8)
        self.assertLessEqual(dst[50, 48], 50)
        self.assertGreaterEqual(dst[50, 51], 200)

    def test_unsharp_mask_float32(self):
        src = np.full((64, 64, 3), 0.5, dtype=np.float32)
        src[:, 32:] = 1.0

        dst = cv.unsharpMask(src, sigma=1.5, amount=2.0)
        self.assertEqual(dst.shape, src.shape)
        self.assertEqual(dst.dtype, np.float32)

    def test_unsharp_mask_threshold(self):
        src = np.full((50, 100), 100, dtype=np.uint8)
        src[0:25, 50:100] = 105
        src[25:50, 50:100] = 200

        dst = cv.unsharpMask(src, sigma=1.0, amount=1.0, threshold=20.0)
        self.assertEqual(dst[10, 48], 100)
        self.assertEqual(dst[10, 52], 105)
        self.assertGreaterEqual(dst[35, 52], 200)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
