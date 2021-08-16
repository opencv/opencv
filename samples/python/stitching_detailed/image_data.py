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
        self.work_imgs = []
        self.seam_imgs = []
        self.compose_imgs = []

        self.work_megapix_scaler = ImageToMegapixScaler(work_megapix)
        self.seam_megapix_scaler = ImageToMegapixScaler(seam_megapix)
        self.compose_megapix_scaler = ImageToMegapixScaler(compose_megapix)

        for img in img_names:
            full_img = ImageData.read_image(img)
            self.work_imgs.append(self.work_megapix_scaler.set_scale_and_downscale(full_img))
            self.seam_imgs.append(self.seam_megapix_scaler.set_scale_and_downscale(full_img))
            self.compose_imgs.append(self.compose_megapix_scaler.set_scale_and_downscale(full_img))

        self.seam_work_aspect = (self.seam_megapix_scaler.scale /
                                 self.work_megapix_scaler.scale)

        self.compose_work_aspect = (self.compose_megapix_scaler.scale /
                                    self.work_megapix_scaler.scale)

    def subset(self, indices):
        self.img_names = Subsetter.subset_list(self.img_names, indices)
        self.seam_imgs = Subsetter.subset_list(self.seam_imgs, indices)
        self.compose_imgs = Subsetter.subset_list(self.compose_imgs, indices)

    def read_image(img_name):
        img = cv.imread(img_name)
        if img is None:
            print("Cannot read image ", img_name)
            exit()
        return img
