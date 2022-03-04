import numpy as np


class MegapixScaler:
    def __init__(self, megapix):
        self.megapix = megapix
        self.is_scale_set = False
        self.scale = None

    def set_scale_by_img_size(self, img_size):
        self.set_scale(
            self.get_scale_by_resolution(img_size[0] * img_size[1])
            )

    def set_scale(self, scale):
        self.scale = scale
        self.is_scale_set = True

    def get_scale_by_resolution(self, resolution):
        if self.megapix > 0:
            return np.sqrt(self.megapix * 1e6 / resolution)
        return 1.0

    def get_scaled_img_size(self, img_size):
        width = int(round(img_size[0] * self.scale))
        height = int(round(img_size[1] * self.scale))
        return (width, height)


class MegapixDownscaler(MegapixScaler):

    @staticmethod
    def force_downscale(scale):
        return min(1.0, scale)

    def set_scale(self, scale):
        scale = self.force_downscale(scale)
        super().set_scale(scale)
