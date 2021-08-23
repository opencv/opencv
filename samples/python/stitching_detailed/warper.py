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

    def set_scale(self, scale):
        self.warper = cv.PyRotationWarper(self.warper_type, scale)

    def warp_images_and_image_masks(self, imgs, cameras, aspect):
        images_warped = []
        masks_warped = []
        corners = []

        for img, camera in zip(imgs, cameras):
            img_warped, mask_warped, corner = self.warp_image_and_image_mask(
                img, camera, aspect
                )
            images_warped.append(img_warped)
            corners.append(corner)
            masks_warped.append(mask_warped)

        return images_warped, masks_warped, corners

    def warp_image_and_image_mask(self, img, camera, aspect):
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

    def warp_roi(self, width, height, camera, aspect=1):
        roi = (width, height)
        K = Warper.get_K(camera, aspect)
        return self.warper.warpRoi(roi, K, camera.R)

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