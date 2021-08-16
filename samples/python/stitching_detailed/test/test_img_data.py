import unittest
import os
import sys

import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..')))

from stitching_detailed.image_data import ImageData
from stitching_detailed.image_to_megapix_scaler import ImageToMegapixScaler
#%%


class TestImageData(unittest.TestCase):

    def test_resizing_of_scaler(self):
        img1 = cv.imread("s1.jpg")
        scaler = ImageToMegapixScaler(0.6)
        resized = scaler.resize_to_scale(img1, scaler.get_scale(img1))

        self.assertEqual(scaler.get_scale(img1), 0.8294067854101966)
        self.assertEqual(resized.shape, (581, 1033, 3))

    def test_force_of_downscaling(self):
        img1 = cv.imread("s1.jpg")
        big_scaler = ImageToMegapixScaler(2)

        normal_scale = big_scaler.get_scale(img1)
        downscale_scale = big_scaler.get_scale_to_force_downscale(img1)

        self.assertEqual(normal_scale, 1.5142826857233715)
        self.assertEqual(downscale_scale, 1.0)

    def test_if_scaler_is_set(self):
        img1, img2 = cv.imread("s1.jpg"), cv.imread("s2.jpg")
        scaler = ImageToMegapixScaler(0.6)
        _ = scaler.set_scale_and_downscale(img1)
        _ = scaler.set_scale_and_downscale(img2)

        self.assertIsNot(scaler.scale, scaler.get_scale(img2))

    def test_image_reading(self):
        image_data = ImageData(["s1.jpg", "s2.jpg"])
        img_shapes = [img.shape for img in image_data.get_full_images()]
        self.assertEqual(img_shapes, [(700, 1246, 3), (700, 1385, 3)])

    def test_image_downscaling(self):
        image_data = ImageData(["s1.jpg", "s2.jpg"])
        img_shapes = [img.shape for img in image_data.get_work_images()]
        self.assertEqual(img_shapes, [(581, 1033, 3), (581, 1149, 3)])

    def test_scaler_aspects(self):
        image_data = ImageData(["s1.jpg", "s2.jpg"])
        _ = image_data.get_seam_images()
        _ = image_data.get_work_images()

        seam_work_aspect = image_data.get_seam_work_aspect()

        self.assertEqual(seam_work_aspect, 0.408248290463863)
        with self.assertRaises(TypeError) as cm:
            image_data.get_compose_work_aspect()
        self.assertTrue(str(cm.exception).startswith("Scale not set yet!"))

    def test_subsetting_indices(self):
        image_data = ImageData(["s1.jpg", "s2.jpg"])
        image_data.update_indices([0])
        img_shapes = [img.shape for img in image_data.get_full_images()]
        self.assertEqual(img_shapes, [(700, 1246, 3)])


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
