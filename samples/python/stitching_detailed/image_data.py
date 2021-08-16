import cv2 as cv

from .image_to_megapix_scaler import ImageToMegapixScaler
from .subsetter import Subsetter


class ImageData:

    DEFAULT_WORK_MEGAPIX = 0.6
    DEFAULT_SEAM_MEGAPIX = 0.1
    DEFAULT_COMPOSE_MEGAPIX = -1

    def __init__(self,
                 img_names,
                 work_megapix=DEFAULT_WORK_MEGAPIX,
                 seam_megapix=DEFAULT_SEAM_MEGAPIX,
                 compose_megapix=DEFAULT_COMPOSE_MEGAPIX):

        self.img_names = img_names
        self.work_megapix_scaler = ImageToMegapixScaler(work_megapix)
        self.seam_megapix_scaler = ImageToMegapixScaler(seam_megapix)
        self.compose_megapix_scaler = ImageToMegapixScaler(compose_megapix)
        self.indices = list(range(len(self.img_names)))

    def get_work_images(self):
        return self.downscale_full_images(self.work_megapix_scaler)

    def get_seam_images(self):
        return self.downscale_full_images(self.seam_megapix_scaler)

    def get_compose_images(self):
        return self.downscale_full_images(self.compose_megapix_scaler)

    def downscale_full_images(self, scaler):
        full_images = self.get_full_images()
        return [scaler.set_scale_and_downscale(img) for img in full_images]

    def get_full_images(self):
        return [self.read_image(img)
                for idx, img in enumerate(self.img_names)
                if idx in self.indices]

    def update_indices(self, indices):
        self.indices = indices

    def get_seam_work_aspect(self):
        return self.__get_scaler_aspect(self.seam_megapix_scaler,
                                        self.work_megapix_scaler)

    def get_compose_work_aspect(self):
        return self.__get_scaler_aspect(self.compose_megapix_scaler,
                                        self.work_megapix_scaler)

    @staticmethod
    def __get_scaler_aspect(scaler1, scaler2):
        return (ImageData.__check_scale(scaler1.scale) /
                ImageData.__check_scale(scaler2.scale))

    @staticmethod
    def __check_scale(scale):
        if scale is not None:
            return scale
        else:
            raise TypeError("Scale not set yet! Have you created the images "
                            "using the get_xxx_imgages() yet?")

    @staticmethod
    def read_image(img_name):
        img = cv.imread(img_name)
        if img is None:
            print("Cannot read image ", img_name)
            exit()
        return img
