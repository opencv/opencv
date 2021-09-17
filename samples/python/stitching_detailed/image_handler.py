import cv2 as cv
from .megapix_downscaler import MegapixDownscaler
from .subsetter import Subsetter


class ImageHandler:

    def __init__(self, medium_megapix, low_megapix, final_megapix):
        self.medium_scaler = MegapixDownscaler(medium_megapix)
        self.low_scaler = MegapixDownscaler(low_megapix)
        self.final_scaler = MegapixDownscaler(final_megapix)
        self.img_names = None
        self.img_sizes = None

    def set_img_names(self, img_names):
        self.img_names = img_names

    def resize_medium_resolution(self):
        return list(self._resize(self.medium_scaler, set_sizes=True))

    def resize_low_resolution(self, imgs):
        return list(self._resize(self.low_scaler, imgs))

    def resize_final_resolution(self):
        return self._resize(self.final_scaler)

    def _resize(self, scaler, imgs=None, set_sizes=False):
        if imgs is None:
            imgs = self.input_images()
        if set_sizes:
            self.img_sizes = []
        for idx, img in enumerate(imgs):
            if set_sizes:
                size = self.get_image_size(img)
                self.img_sizes.append(size)
            if not scaler.is_scale_set:
                scaler.set_scale_by_img_size(self.img_sizes[0])
            dsize = scaler.get_scaled_img_size(self.img_sizes[idx])
            yield cv.resize(img, dsize, interpolation=cv.INTER_LINEAR_EXACT)

    @staticmethod
    def get_image_size(img):
        """(width, height)"""
        return (img.shape[1], img.shape[0])

    def input_images(self):
        for name in self.img_names:
            img = self.read_image(name)
            yield img

    @staticmethod
    def read_image(img_name):
        img = cv.imread(img_name)
        if img is None:
            print("Cannot read image ", img_name)
            exit()
        return img

    def subset(self, indices):
        self.img_names = Subsetter.subset_list(self.img_names, indices)
        self.img_sizes = Subsetter.subset_list(self.img_sizes, indices)

    def get_compose_work_aspect(self):
        return self.final_scaler.get_aspect_to(self.medium_scaler)

    def get_seam_work_aspect(self):
        return self.low_scaler.get_aspect_to(self.medium_scaler)

    def get_seam_compose_aspect(self):
        return self.low_scaler.get_aspect_to(self.final_scaler)
