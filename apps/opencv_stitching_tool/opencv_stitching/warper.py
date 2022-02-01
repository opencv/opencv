from statistics import median

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

    def __init__(self, warper_type=DEFAULT_WARP_TYPE):
        self.warper_type = warper_type
        self.scale = None

    def set_scale(self, cameras):
        focals = [cam.focal for cam in cameras]
        self.scale = median(focals)

    def warp_images(self, imgs, cameras, aspect=1):
        for img, camera in zip(imgs, cameras):
            yield self.warp_image(img, camera, aspect)

    def warp_image(self, img, camera, aspect=1):
        warper = cv.PyRotationWarper(self.warper_type, self.scale*aspect)
        _, warped_image = warper.warp(img,
                                      Warper.get_K(camera, aspect),
                                      camera.R,
                                      cv.INTER_LINEAR,
                                      cv.BORDER_REFLECT)
        return warped_image

    def create_and_warp_masks(self, sizes, cameras, aspect=1):
        for size, camera in zip(sizes, cameras):
            yield self.create_and_warp_mask(size, camera, aspect)

    def create_and_warp_mask(self, size, camera, aspect=1):
        warper = cv.PyRotationWarper(self.warper_type, self.scale*aspect)
        mask = 255 * np.ones((size[1], size[0]), np.uint8)
        _, warped_mask = warper.warp(mask,
                                     Warper.get_K(camera, aspect),
                                     camera.R,
                                     cv.INTER_NEAREST,
                                     cv.BORDER_CONSTANT)
        return warped_mask

    def warp_rois(self, sizes, cameras, aspect=1):
        roi_corners = []
        roi_sizes = []
        for size, camera in zip(sizes, cameras):
            roi = self.warp_roi(size, camera, aspect)
            roi_corners.append(roi[0:2])
            roi_sizes.append(roi[2:4])
        return roi_corners, roi_sizes

    def warp_roi(self, size, camera, aspect=1):
        warper = cv.PyRotationWarper(self.warper_type, self.scale*aspect)
        K = Warper.get_K(camera, aspect)
        return warper.warpRoi(size, K, camera.R)

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
