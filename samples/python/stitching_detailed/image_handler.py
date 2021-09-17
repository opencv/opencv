import cv2 as cv
from .megapix_downscaler import MegapixDownscaler
from .subsetter import Subsetter


class ImageHandler:

    DEFAULT_MEDIUM_MEGAPIX = 0.6
    DEFAULT_LOW_MEGAPIX = 0.1
    DEFAULT_FINAL_MEGAPIX = -1

    def __init__(self,
                 medium_megapix=DEFAULT_MEDIUM_MEGAPIX,
                 low_megapix=DEFAULT_LOW_MEGAPIX,
                 final_megapix=DEFAULT_FINAL_MEGAPIX):

        if medium_megapix <= low_megapix:
            raise ValueError("Medium resolution megapix need to be greater or "
                             "equal than low resolution megapix")

        self.medium_scaler = MegapixDownscaler(medium_megapix)
        self.low_scaler = MegapixDownscaler(low_megapix)
        self.final_scaler = MegapixDownscaler(final_megapix)

        self.scales_set = False
        self.img_names = []
        self.img_sizes = []

    def set_img_names(self, img_names):
        self.img_names = img_names

    def resize_to_medium_resolution(self):
        return self._read_and_resize_imgs(self.medium_scaler)

    def resize_to_low_resolution(self, medium_imgs=None):
        if medium_imgs and self.scales_set:
            return self._resize_medium_to_low(medium_imgs)
        return self._read_and_resize_imgs(self.low_scaler)

    def resize_to_final_resolution(self):
        return self._read_and_resize_imgs(self.final_scaler)

    def _read_and_resize_imgs(self, scaler):
        for img, size in self._input_images():
            yield self._resize_img_by_scaler(scaler, size, img)

    def _resize_medium_to_low(self, medium_imgs):
        for img, size in zip(medium_imgs, self.img_sizes):
            yield self._resize_img_by_scaler(self.low_scaler, size, img)

    @staticmethod
    def _resize_img_by_scaler(scaler, size, img):
        desired_size = scaler.get_scaled_img_size(size)
        return cv.resize(img, desired_size,
                         interpolation=cv.INTER_LINEAR_EXACT)

    def _input_images(self):
        self.img_sizes = []
        for name in self.img_names:
            img = self._read_image(name)
            size = self._get_image_size(img)
            self.img_sizes.append(size)
            self._set_scaler_scales()
            yield img, size

    @staticmethod
    def _get_image_size(img):
        """(width, height)"""
        return (img.shape[1], img.shape[0])

    @staticmethod
    def _read_image(img_name):
        img = cv.imread(img_name)
        if img is None:
            print("Cannot read image ", img_name)
            exit()
        return img

    def _set_scaler_scales(self):
        if not self.scales_set:
            first_img_size = self.img_sizes[0]
            self.medium_scaler.set_scale_by_img_size(first_img_size)
            self.low_scaler.set_scale_by_img_size(first_img_size)
            self.final_scaler.set_scale_by_img_size(first_img_size)
        self.scales_set = True

    def subset(self, indices):
        self.img_names = Subsetter.subset_list(self.img_names, indices)
        self.img_sizes = Subsetter.subset_list(self.img_sizes, indices)

    def get_compose_work_aspect(self):
        return self.final_scaler.get_aspect_to(self.medium_scaler)

    def get_seam_work_aspect(self):
        return self.low_scaler.get_aspect_to(self.medium_scaler)

    def get_seam_compose_aspect(self):
        return self.low_scaler.get_aspect_to(self.final_scaler)
