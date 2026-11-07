#!/usr/bin/env python
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests


class warp_border_values_test(NewOpenCVTests):

    def test_per_channel_fill(self):
        affine = np.array([[1, 0, 137], [0, 1, 137]], dtype=np.float32)
        perspective = np.vstack([affine, [0, 0, 1]]).astype(np.float32)
        mapx = np.full((5, 7), -137, dtype=np.float32)
        mapy = mapx.copy()
        for dtype in (np.uint8, np.float16, np.float32, np.float64):
            image = np.zeros((5, 7, 20), dtype=dtype)
            holder = np.empty((20, 2), dtype=np.float64)
            holder[:, 0] = np.arange(20) if dtype == np.uint8 else np.arange(20)/32.
            values = holder[:, 0]
            expected = np.broadcast_to(values.astype(dtype), image.shape)
            for interpolation in (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_LANCZOS4):
                if dtype == np.float16 and interpolation == cv.INTER_NEAREST:
                    continue
                results = (
                    cv.warpAffineWithBorderValues(image, affine, (7, 5), values, flags=interpolation),
                    cv.warpAffineWithBorderValues(image, affine, (7, 5), tuple(values), flags=interpolation),
                    cv.warpPerspectiveWithBorderValues(image, perspective, (7, 5), values, flags=interpolation),
                    cv.remapWithBorderValues(image, mapx, mapy, interpolation, borderValues=values),
                )
                for result in results:
                    self.assertEqual(result.dtype, image.dtype)
                    np.testing.assert_array_equal(result, expected)

    def test_strided_arrays(self):
        base = np.arange(20*17*13, dtype=np.uint8).reshape(17, 13, 20)
        image = base[1::2, 2::2]
        height, width = image.shape[:2]
        values = np.arange(20, dtype=np.float64)
        expected = np.broadcast_to(values.astype(np.uint8), image.shape).copy()
        expected[1:, 1:] = image[:-1, :-1]
        holder = np.empty((height + 3, width + 4, 20), dtype=np.uint8)
        destination = holder[1:height + 1, 2:width + 2]
        matrix = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
        result = cv.warpAffineWithBorderValues(
            image, matrix, (width, height), values, dst=destination, flags=cv.INTER_NEAREST
        )
        self.assertTrue(np.shares_memory(result, destination))
        np.testing.assert_array_equal(destination, expected)

    def test_invalid_length(self):
        image = np.zeros((5, 7, 20), dtype=np.float32)
        values = np.arange(19, dtype=np.float64)
        matrix = np.eye(3, dtype=np.float32)
        mapx = np.zeros((5, 7), dtype=np.float32)
        with self.assertRaises(cv.error):
            cv.warpAffineWithBorderValues(image, matrix[:2], (7, 5), values)
        with self.assertRaises(cv.error):
            cv.warpPerspectiveWithBorderValues(image, matrix, (7, 5), values)
        with self.assertRaises(cv.error):
            cv.remapWithBorderValues(image, mapx, mapx, cv.INTER_LINEAR, values)

    def test_original_scalar_argument(self):
        image = np.ones((5, 7, 3), dtype=np.float32)
        matrix = np.array([[1, 0, 137], [0, 1, 137]], dtype=np.float32)
        expected = np.zeros_like(image)
        expected[..., 0] = 0.5
        result = cv.warpAffine(image, matrix, (7, 5), borderValue=0.5)
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
