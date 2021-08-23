import cv2 as cv
import numpy as np


class ImageToMegapixScaler:
    def __init__(self, megapix):
        self.megapix = megapix
        self.is_scale_set = False
        self.scale = None

    def get_scale_by_resolution(self, resolution):
        if self.megapix > 0:
            return np.sqrt(self.megapix * 1e6 / resolution)
        else:
            return 1.0

    def get_scale_by_image(self, img):
        return self.get_scale_by_resolution(self.get_image_resolution(img))

    @staticmethod
    def get_image_resolution(img):
        return img.shape[0] * img.shape[1]

    @staticmethod
    def resize_to_scale(img, scale):
        if scale != 1.0:
            return cv.resize(src=img, dsize=None,
                             fx=scale, fy=scale,
                             interpolation=cv.INTER_LINEAR_EXACT)
        else:
            return img

    def set_scale_if_not_set(self, scale):
        if self.is_scale_set is False:
            self.scale = scale
            self.is_scale_set = True

    def resize_to_set_scale(self, img):
        if self.is_scale_set:
            return self.resize_to_scale(img, self.scale)
        else:
            print("Scale not set")
            exit()

    @staticmethod
    def force_downscale(scale):
        return min(1.0, scale)

    def set_scale_if_not_set_and_downscale(self, img):
        self.set_scale_if_not_set(
            self.force_downscale(
                self.get_scale_by_image(img)
                )
            )
        return self.resize_to_set_scale(img)

    def set_downscale_scale_by_resolution(self, resolution):
        self.scale = self.force_downscale(
            self.get_scale_by_resolution(resolution)
            )
        self.is_scale_set = True

    def get_aspect_to(self, scaler):
        if self.is_scale_set and scaler.is_scale_set:
            return self.scale / scaler.scale
        else:
            print("Scale not set")
            exit()

