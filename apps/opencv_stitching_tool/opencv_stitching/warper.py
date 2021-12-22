import cv2 as cv
import numpy as np


class Warper:

    WARP_TYPE_CHOICES = ('spherical', 'plane', 'affine', 'cylindrical',
                         'fisheye', 'stereographic', 'compressedPlaneA2B1',
                         'compressedPlaneA1.5B1',
                         'compressedPlanePortraitA2B1',
                         'compressedPlanePortraitA1.5B1',
                         'paniniA2B1', 'paniniA1.5B1', 'paniniPortraitA2B1',
                         'paniniPortraitA1.5B1', 'mercator',
                         'transverseMercator')

    DEFAULT_WARP_TYPE = 'spherical'

    def __init__(self, warper_type=DEFAULT_WARP_TYPE, scale=1):
        self.warper_type = warper_type
        self.warper = cv.PyRotationWarper(warper_type, scale)
        self.scale = scale

    def warp_images_and_image_masks(self, imgs, cameras, scale=None, aspect=1):
        self.update_scale(scale)
        for img, camera in zip(imgs, cameras):
            yield self.warp_image_and_image_mask(img, camera, scale, aspect)

    def warp_image_and_image_mask(self, img, camera, scale=None, aspect=1):
        self.update_scale(scale)
        corner, img_warped = self.warp_image(img, camera, aspect)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        _, mask_warped = self.warp_image(mask, camera, aspect, mask=True)
        return img_warped, mask_warped, corner

    def warp_image(self, image, camera, aspect=1, mask=False):
        if mask:
            interp_mode = cv.INTER_NEAREST
            border_mode = cv.BORDER_CONSTANT
        else:
            interp_mode = cv.INTER_LINEAR
            border_mode = cv.BORDER_REFLECT

        corner, warped_image = self.warper.warp(image,
                                                Warper.get_K(camera, aspect),
                                                camera.R,
                                                interp_mode,
                                                border_mode)
        return corner, warped_image

    def warp_roi(self, width, height, camera, scale=None, aspect=1):
        self.update_scale(scale)
        roi = (width, height)
        K = Warper.get_K(camera, aspect)
        return self.warper.warpRoi(roi, K, camera.R)

    def update_scale(self, scale):
        if scale is not None and scale != self.scale:
            self.warper = cv.PyRotationWarper(self.warper_type, scale)  # setScale not working: https://docs.opencv.org/4.x/d5/d76/classcv_1_1PyRotationWarper.html#a90b000bb75f95294f9b0b6ec9859eb55
            self.scale = scale

    @staticmethod
    def get_K(camera, aspect=1):
        K = camera.K().astype(np.float32)
        """ Modification of intrinsic parameters needed if cameras were
        obtained on different scale than the scale of the Images which should
        be warped """
        K[0, 0] *= aspect
        K[0, 2] *= aspect
        K[1, 1] *= aspect
        K[1, 2] *= aspect
        return K
