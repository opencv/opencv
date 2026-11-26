import cv2 as cv
import numpy as np
from tests_common import NewOpenCVTests

class Resize3DTest(NewOpenCVTests):
    def test_resize3D_basic_uint8(self):
        src = np.arange(2*4*4*3, dtype=np.uint8).reshape(2, 4, 4, 3)
        dst1 = cv.resize3D(src, (4, 8, 8), interpolation=cv.INTER_NEAREST)
        self.assertEqual(dst1.shape, (4, 8, 8, 3))
        self.assertEqual(dst1.dtype, np.uint8)
        self.assertEqual(dst1[0, 0, 0, 0], src[0, 0, 0, 0])

        dst2 = cv.resize3d(src, (4, 8, 8), interpolation=cv.INTER_NEAREST)
        self.assertEqual(dst2.shape, (4, 8, 8, 3))

    def test_resize3D_float32_linear(self):
        src = np.ones((2, 4, 4, 5), dtype=np.float32) * 10.0
        dst = cv.resize3D(src, (4, 8, 8), interpolation=cv.INTER_LINEAR)
        self.assertEqual(dst.shape, (4, 8, 8, 5))
        self.assertEqual(dst.dtype, np.float32)
        np.testing.assert_allclose(dst, 10.0, rtol=1e-5)

    def test_resize3D_large_channels(self):
        src = np.random.randn(2, 4, 4, 576).astype(np.float32)
        dst = cv.resize3D(src, (4, 8, 8), interpolation=cv.INTER_LINEAR)
        self.assertEqual(dst.shape, (4, 8, 8, 576))

    def test_resize3D_unit_length_axis(self):
        src = np.ones((4, 8, 8, 3), dtype=np.uint8) * 128
        dst = cv.resize3D(src, (1, 4, 4), interpolation=cv.INTER_LINEAR)
        self.assertEqual(dst.shape, (1, 4, 4, 3))
        self.assertTrue(np.all(dst == 128))

    def test_resize3D_scale_factors(self):
        src = np.ones((2, 4, 4, 2), dtype=np.float32)
        dst = cv.resize3D(src, (0, 0, 0), fx=2.0, fy=2.0, fz=2.0, interpolation=cv.INTER_LINEAR)
        self.assertEqual(dst.shape, (4, 8, 8, 2))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
