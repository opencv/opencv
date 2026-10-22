#!/usr/bin/env python

import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests


class input_array_typing_test(NewOpenCVTests):
    def test_scalar_operands(self):
        image = np.ones((3, 5, 3), dtype=np.float32)
        scalar = (1.0, 2.0, 3.0, 0.0)
        added = cv.add(image, scalar)
        multiplied = cv.multiply(image, scalar)
        np.testing.assert_array_equal(added, image + np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(multiplied, image * np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_min_max_scalar_operands(self):
        gray = np.arange(15, dtype=np.float32).reshape(3, 5)
        color = np.arange(45, dtype=np.float32).reshape(3, 5, 3)
        scalar = (1.0, 2.0, 3.0, 0.0)
        for operation, reference in ((cv.min, np.minimum), (cv.max, np.maximum)):
            for image, value, expected in (
                (gray, 2.5, reference(gray, 2.5)),
                (color, scalar, reference(color, np.array(scalar[:3], dtype=np.float32))),
            ):
                destination = np.empty_like(image)
                for first, second in ((image, value), (value, image)):
                    self.assertIs(operation(first, second, dst=destination), destination)
                    np.testing.assert_array_equal(destination, expected)

    def test_min_max_umat_scalar_operands(self):
        for operation, reference in ((cv.min, np.minimum), (cv.max, np.maximum)):
            for image_type, image_value, scalar in (
                (cv.CV_32FC1, 4.0, 2.5),
                (cv.CV_32FC3, (1.0, 3.0, 5.0, 0.0), (2.0, 4.0, 0.0, 0.0)),
            ):
                image = cv.UMat(3, 5, image_type, image_value)
                array = image.get()
                values = scalar[:3] if isinstance(scalar, tuple) else scalar
                expected = reference(array, np.array(values, dtype=np.float32))
                for result in (operation(image, scalar), operation(scalar, image)):
                    self.assertIsInstance(result, cv.UMat)
                    np.testing.assert_array_equal(result.get(), expected)

    def test_umat_numpy_constructors(self):
        for flat in (np.arange(45, dtype=np.uint8), np.arange(45, dtype=np.float32)):
            for channels in (1, 3):
                shape = (3, 5) if channels == 1 else (3, 5, channels)
                image = flat[:15 * channels].reshape(shape)
                original = image.copy()
                for actual, expected in (
                    (cv.UMat(image), original),
                    (cv.UMat(m=image), original),
                    (cv.UMat(image, (0, 2), (0, 3)), original[:2, :3]),
                    (cv.UMat(image, (1, 1, 3, 2)), original[1:3, 1:4]),
                    (cv.UMat(image, [(0, 2), (0, 3)]), original[:2, :3]),
                ):
                    result = actual.get()
                    np.testing.assert_array_equal(result, expected)
                    self.assertEqual(result.dtype, original.dtype)
                np.testing.assert_array_equal(image, original)

    def test_empty_second_remap_map(self):
        image = np.arange(15, dtype=np.uint8).reshape(3, 5)
        x, y = np.meshgrid(np.arange(5, dtype=np.float32), np.arange(3, dtype=np.float32))
        map1, _ = cv.convertMaps(x, y, cv.CV_16SC2, nninterpolation=True)
        remapped = cv.remap(image, map1, None, cv.INTER_NEAREST)
        np.testing.assert_array_equal(remapped, image)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
