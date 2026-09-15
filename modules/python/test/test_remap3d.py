#!/usr/bin/env python

import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests


class Remap3DTest(NewOpenCVTests):
    def test_volume_and_label_map(self):
        z, y, x = np.indices((3, 4, 5), dtype=np.float32)
        volume = cv.Mat(100 * z + 10 * y + x, wrap_channels=False)
        coordinates = cv.Mat(np.stack((x, y, z + 0.25), axis=-1), wrap_channels=True)
        result = cv.remap3D(volume, coordinates, cv.INTER_LINEAR,
                            borderMode=cv.BORDER_REPLICATE)
        expected = 100 * np.minimum(z + 0.25, 2) + 10 * y + x
        self.assertEqual(result.shape, volume.shape)
        np.testing.assert_allclose(result, expected)

        labels = cv.Mat(z.astype(np.uint8), wrap_channels=False)
        result = cv.remap3D(labels, coordinates, cv.INTER_NEAREST,
                            borderMode=cv.BORDER_REPLICATE)
        np.testing.assert_array_equal(result, labels)

    def test_multichannel_identity(self):
        z, y, x = np.indices((2, 3, 4), dtype=np.float32)
        coordinates = cv.Mat(np.stack((x, y, z), axis=-1), wrap_channels=True)
        for dtype in (np.uint8, np.float32):
            for channels in (2, 3, 4):
                data = np.arange(2 * 3 * 4 * channels, dtype=dtype).reshape(2, 3, 4, channels)
                volume = cv.Mat(data, wrap_channels=True)
                for interpolation in (cv.INTER_NEAREST, cv.INTER_LINEAR):
                    result = cv.remap3D(volume, coordinates, interpolation)
                    np.testing.assert_array_equal(result, volume)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
