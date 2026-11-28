#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests


def make_image(size, channels, seed):
    """Gradient plus a shape, so the codecs see something more realistic than noise."""
    h, w = size
    gx = np.tile(np.linspace(0, 255, w, dtype=np.float32), (h, 1))
    gy = np.tile(np.linspace(0, 255, h, dtype=np.float32).reshape(-1, 1), (1, w))
    base = ((gx + gy) / 2).astype(np.uint8)

    if channels == 1:
        img = base
    else:
        img = np.dstack([base, (base.astype(np.int32) + 85) % 256, (base.astype(np.int32) + 170) % 256])
        img = img.astype(np.uint8)
        if channels == 4:
            # Alpha stays >= 1: libwebp may discard the colour of fully transparent pixels.
            img = np.dstack([img, np.maximum(gx, 1).astype(np.uint8)])

    img = np.ascontiguousarray(img)
    cv.circle(img, (w // 3 + seed % 5, h // 2), max(3, min(w, h) // 5), (0, 0, 0, 255)[:channels], -1)
    return img


def make_batch(count, size=(48, 64), channels=3):
    return [make_image(size, channels, i) for i in range(count)]


class imgcodecs_batch_test(NewOpenCVTests):

    def test_issue_example_shape(self):
        """The exact call shape requested in opencv/opencv#29587."""
        images = make_batch(4)

        ok, buffers = cv.imencodeBatch(".jpg", images, [cv.IMWRITE_JPEG_QUALITY, 90])
        self.assertTrue(ok)
        self.assertEqual(len(buffers), len(images))
        for buf in buffers:
            self.assertIsInstance(buf, np.ndarray)
            self.assertEqual(buf.dtype, np.uint8)
            self.assertGreater(buf.size, 0)

        ok, decoded = cv.imdecodeBatch(buffers, cv.IMREAD_UNCHANGED)
        self.assertTrue(ok)
        self.assertEqual(len(decoded), len(images))
        for src, dst in zip(images, decoded):
            self.assertEqual(src.shape, dst.shape)

    def test_encode_matches_python_loop(self):
        """The whole point of the feature: identical bytes, without looping in Python."""
        for ext, params in ((".png", []), (".jpg", [cv.IMWRITE_JPEG_QUALITY, 90]),
                            (".webp", [cv.IMWRITE_WEBP_QUALITY, 90])):
            if not cv.haveImageWriter("test" + ext):
                continue
            images = make_batch(5)

            ok, batched = cv.imencodeBatch(ext, images, params)
            self.assertTrue(ok, ext)
            self.assertEqual(len(batched), len(images), ext)

            for i, img in enumerate(images):
                ok, single = cv.imencode(ext, img, params)
                self.assertTrue(ok, ext)
                self.assertTrue(np.array_equal(single, batched[i]), "%s #%d" % (ext, i))

    def test_png_roundtrip_is_lossless(self):
        if not cv.haveImageWriter("test.png"):
            return
        images = make_batch(5)

        ok, buffers = cv.imencodeBatch(".png", images, [])
        self.assertTrue(ok)

        ok, decoded = cv.imdecodeBatch(buffers, cv.IMREAD_COLOR)
        self.assertTrue(ok)
        for i, (src, dst) in enumerate(zip(images, decoded)):
            self.assertTrue(np.array_equal(src, dst), "#%d" % i)

    def test_order_is_preserved(self):
        if not cv.haveImageWriter("test.png"):
            return
        # Distinct flat colors: any reordering shows up immediately.
        images = [np.full((16, 16, 3), v, dtype=np.uint8) for v in (10, 20, 30, 40, 50, 60, 70, 80)]

        ok, buffers = cv.imencodeBatch(".png", images, [])
        self.assertTrue(ok)
        ok, decoded = cv.imdecodeBatch(buffers, cv.IMREAD_COLOR)
        self.assertTrue(ok)

        for i, img in enumerate(images):
            self.assertEqual(int(decoded[i][0, 0, 0]), int(img[0, 0, 0]), "#%d" % i)

    def test_mixed_sizes(self):
        if not cv.haveImageWriter("test.png"):
            return
        images = [make_image((5, 17), 3, 0), make_image((64, 64), 3, 1), make_image((33, 128), 3, 2)]

        ok, buffers = cv.imencodeBatch(".png", images, [])
        self.assertTrue(ok)
        ok, decoded = cv.imdecodeBatch(buffers, cv.IMREAD_COLOR)
        self.assertTrue(ok)

        for i, (src, dst) in enumerate(zip(images, decoded)):
            self.assertEqual(src.shape, dst.shape, "#%d" % i)
            self.assertTrue(np.array_equal(src, dst), "#%d" % i)

    def test_grayscale_and_alpha(self):
        if not cv.haveImageWriter("test.png"):
            return
        for channels in (1, 3, 4):
            images = make_batch(3, channels=channels)

            ok, buffers = cv.imencodeBatch(".png", images, [])
            self.assertTrue(ok, "channels=%d" % channels)
            ok, decoded = cv.imdecodeBatch(buffers, cv.IMREAD_UNCHANGED)
            self.assertTrue(ok, "channels=%d" % channels)

            for i, (src, dst) in enumerate(zip(images, decoded)):
                self.assertTrue(np.array_equal(src, dst), "channels=%d #%d" % (channels, i))

    def test_empty_batch(self):
        ok, buffers = cv.imencodeBatch(".png", [], [])
        self.assertTrue(ok)
        self.assertEqual(len(buffers), 0)

        ok, decoded = cv.imdecodeBatch([], cv.IMREAD_COLOR)
        self.assertTrue(ok)
        self.assertEqual(len(decoded), 0)

    def test_failure_keeps_batch_length(self):
        """A bad item must be reported in place, not silently dropped."""
        if not cv.haveImageWriter("test.png"):
            return
        images = make_batch(4)

        ok, buffers = cv.imencodeBatch(".png", images, [])
        self.assertTrue(ok)

        buffers = list(buffers)
        buffers[1] = np.full(64, 127, dtype=np.uint8)  # unrecognizable signature

        ok, decoded = cv.imdecodeBatch(buffers, cv.IMREAD_COLOR)
        self.assertFalse(ok)
        self.assertEqual(len(decoded), len(images))
        # The slot is kept so the indices still line up; an empty Mat surfaces as None in Python.
        self.assertIsNone(decoded[1])
        self.assertTrue(np.array_equal(images[0], decoded[0]))
        self.assertTrue(np.array_equal(images[2], decoded[2]))

    def test_unknown_extension_raises(self):
        images = make_batch(2)
        with self.assertRaises(cv.error):
            cv.imencodeBatch(".notaformat", images, [])


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
