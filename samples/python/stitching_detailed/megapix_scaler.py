import cv2 as cv
import numpy as np


class MegapixScaler:
    def __init__(self, megapix):
        self.megapix = megapix
        self.is_scale_set = False
        self.scale = None

    def set_scale_by_img_size(self, img_size):
        self._set_scale(
            self._get_scale_by_resolution(img_size[0] * img_size[1])
            )

    def _get_scale_by_resolution(self, resolution):
        if self.megapix > 0:
            return np.sqrt(self.megapix * 1e6 / resolution)
        else:
            return 1.0

    def _set_scale(self, scale):
        self.scale = scale
        self.is_scale_set = True

    def resize(self, img):
        if self.is_scale_set:
            return self._resize_to_scale(img, self.scale)
        else:
            print("Scale not set")
            exit()

    @staticmethod
    def _resize_to_scale(img, scale):
        if scale != 1.0:
            return cv.resize(src=img, dsize=None,
                             fx=scale, fy=scale,
                             interpolation=cv.INTER_LINEAR_EXACT)
        else:
            return img

    def get_aspect_to(self, scaler):
        if self.is_scale_set and scaler.is_scale_set:
            return self.scale / scaler.scale
        else:
            print("Scale not set")
            exit()
