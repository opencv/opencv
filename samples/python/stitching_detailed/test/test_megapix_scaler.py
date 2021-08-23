import unittest
import os
import sys

import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..')))

from stitching_detailed.image_to_megapix_scaler import ImageToMegapixScaler
#%%


class TestImageData(unittest.TestCase):

    def test_get_scale_by_resolution(self):
        scaler = ImageToMegapixScaler(0.6)

        scale = scaler.get_scale_by_resolution(1_200_000)

        self.assertEqual(scale, 0.7071067811865476)

    def test_get_scale_by_image(self):
        img1 = cv.imread("s1.jpg")
        scaler = ImageToMegapixScaler(0.6)

        scale = scaler.get_scale_by_image(img1)

        self.assertEqual(scale, 0.8294067854101966)

    def test_resizing_of_scaler(self):
        img1 = cv.imread("s1.jpg")
        scaler = ImageToMegapixScaler(0.6)

        resized = scaler.resize_to_scale(img1, scaler.get_scale_by_image(img1))

        self.assertEqual(resized.shape, (581, 1033, 3))
        # 581*1033 = 600173 px = ~0.6 MP

    def test_force_of_downscaling(self):
        img1 = cv.imread("s1.jpg")
        big_scaler = ImageToMegapixScaler(2)

        normal_scale = big_scaler.get_scale_by_image(img1)
        downscale_scale = big_scaler.force_downscale(normal_scale)

        self.assertEqual(normal_scale, 1.5142826857233715)
        self.assertEqual(downscale_scale, 1.0)

    def test_if_scaler_is_set(self):
        img1, img2 = cv.imread("s1.jpg"), cv.imread("s2.jpg")
        scaler = ImageToMegapixScaler(0.6)
        _ = scaler.set_scale_if_not_set_and_downscale(img1)
        _ = scaler.set_scale_if_not_set_and_downscale(img2)

        scaler_scale = scaler.scale
        scale_img2 = scaler.get_scale_by_image(img2)

        self.assertIsNot(scaler_scale, scale_img2)

    def test_set_scale_to_downscale_resolution(self):
        scaler = ImageToMegapixScaler(0.6)
        scaler.set_downscale_scale_by_resolution(1246 * 700)
        scale = scaler.scale

        self.assertEqual(scale, 0.8294067854101966)

    def test_get_aspect(self):
        scaler1 = ImageToMegapixScaler(1)
        scaler1.set_scale_if_not_set(0.1)
        scaler2 = ImageToMegapixScaler(1)
        scaler2.set_scale_if_not_set(0.6)

        aspect = scaler1.get_aspect_to(scaler2)

        self.assertEqual(aspect, 0.16666666666666669)

def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
