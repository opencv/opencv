import cv2 as cv

from .megapix_scaler import MegapixDownscaler
from .stitching_error import StitchingError

class ImageHandler:

    DEFAULT_MEDIUM_MEGAPIX = 0.6
    DEFAULT_LOW_MEGAPIX = 0.1
    DEFAULT_FINAL_MEGAPIX = -1

    def __init__(self,
                 medium_megapix=DEFAULT_MEDIUM_MEGAPIX,
                 low_megapix=DEFAULT_LOW_MEGAPIX,
                 final_megapix=DEFAULT_FINAL_MEGAPIX):

        if medium_megapix < low_megapix:
            raise StitchingError("Medium resolution megapix need to be "
                                 "greater or equal than low resolution "
                                 "megapix")

        self.medium_scaler = MegapixDownscaler(medium_megapix)
        self.low_scaler = MegapixDownscaler(low_megapix)
        self.final_scaler = MegapixDownscaler(final_megapix)

        self.scales_set = False
        self.img_names = []
        self.img_sizes = []

    def set_img_names(self, img_names):
        self.img_names = img_names

    def resize_to_medium_resolution(self):
        return self.read_and_resize_imgs(self.medium_scaler)

    def resize_to_low_resolution(self, medium_imgs=None):
        if medium_imgs and self.scales_set:
            return self.resize_imgs_by_scaler(medium_imgs, self.low_scaler)
        return self.read_and_resize_imgs(self.low_scaler)

    def resize_to_final_resolution(self):
        return self.read_and_resize_imgs(self.final_scaler)

    def read_and_resize_imgs(self, scaler):
        for img, size in self.input_images():
            yield self.resize_img_by_scaler(scaler, size, img)

    def resize_imgs_by_scaler(self, medium_imgs, scaler):
        for img, size in zip(medium_imgs, self.img_sizes):
            yield self.resize_img_by_scaler(scaler, size, img)

    @staticmethod
    def resize_img_by_scaler(scaler, size, img):
        desired_size = scaler.get_scaled_img_size(size)
        return cv.resize(img, desired_size,
                         interpolation=cv.INTER_LINEAR_EXACT)

    def input_images(self):
        self.img_sizes = []
        for name in self.img_names:
            img = self.read_image(name)
            size = self.get_image_size(img)
            self.img_sizes.append(size)
            self.set_scaler_scales()
            yield img, size

    @staticmethod
    def get_image_size(img):
        """(width, height)"""
        return (img.shape[1], img.shape[0])

    @staticmethod
    def read_image(img_name):
        img = cv.imread(img_name)
        if img is None:
            raise StitchingError("Cannot read image " + img_name)
        return img

    def set_scaler_scales(self):
        if not self.scales_set:
            first_img_size = self.img_sizes[0]
            self.medium_scaler.set_scale_by_img_size(first_img_size)
            self.low_scaler.set_scale_by_img_size(first_img_size)
            self.final_scaler.set_scale_by_img_size(first_img_size)
        self.scales_set = True

    def get_medium_to_final_ratio(self):
        return self.final_scaler.scale / self.medium_scaler.scale

    def get_medium_to_low_ratio(self):
        return self.low_scaler.scale / self.medium_scaler.scale

    def get_final_to_low_ratio(self):
        return self.low_scaler.scale / self.final_scaler.scale

    def get_low_to_final_ratio(self):
        return self.final_scaler.scale / self.low_scaler.scale

    def get_final_img_sizes(self):
        return [self.final_scaler.get_scaled_img_size(sz)
                for sz in self.img_sizes]

    def get_low_img_sizes(self):
        return [self.low_scaler.get_scaled_img_size(sz)
                for sz in self.img_sizes]
