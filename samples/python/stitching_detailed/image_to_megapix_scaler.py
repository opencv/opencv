import cv2 as cv
import numpy as np


class ImageToMegapixScaler:
    def __init__(self, megapix):
        self.megapix = megapix
        self.is_scale_set = False
        self.scale = None

    def get_scale(self, img):
        if self.megapix > 0:
            return np.sqrt(self.megapix * 1e6 /
                           (img.shape[0] * img.shape[1]))
        else:
            return 1.0

    def get_scale_to_force_downscale(self, img):
        return min(1.0, self.get_scale(img))

    def resize_to_scale(self, img, scale):
        return cv.resize(src=img, dsize=None,
                         fx=scale, fy=scale,
                         interpolation=cv.INTER_LINEAR_EXACT)

    def set_scale_and_downscale(self, img):
        if self.is_scale_set is False:
            self.scale = self.get_scale_to_force_downscale(img)
            self.is_scale_set = True
        return self.resize_to_scale(img, self.scale)
