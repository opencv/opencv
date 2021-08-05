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
        self.warper = cv.PyRotationWarper(warper_type, scale)

    def warp_image(self, image, camera, aspect=1):

        corner, warped_image = self.warper.warp(image,
                                                Warper.get_K(camera, aspect),
                                                camera.R,
                                                cv.INTER_LINEAR,
                                                cv.BORDER_REFLECT)
        return corner, warped_image

    # def warp_images(self, images, cameras, aspect=1):
    #     for image, camera in zip(images, cameras):
    #         self.warp_image(image, camera, aspect)

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
