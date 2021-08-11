from collections import OrderedDict
import cv2 as cv
import numpy as np


class SeamFinder:
    """https://docs.opencv.org/master/d7/d09/classcv_1_1detail_1_1SeamFinder.html"""  # noqa
    SEAM_FINDER_CHOICES = OrderedDict()
    SEAM_FINDER_CHOICES['dp_color'] = cv.detail_DpSeamFinder('COLOR')
    SEAM_FINDER_CHOICES['dp_colorgrad'] = cv.detail_DpSeamFinder('COLOR_GRAD')
    SEAM_FINDER_CHOICES['voronoi'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)  # noqa
    SEAM_FINDER_CHOICES['no'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)  # noqa

    DEFAULT_SEAM_FINDER = list(SEAM_FINDER_CHOICES.keys())[0]

    def __init__(self, finder=DEFAULT_SEAM_FINDER):
        self.finder = SeamFinder.SEAM_FINDER_CHOICES[finder]

    def find(self, imgs, corners, masks):
        """https://docs.opencv.org/master/d0/dd5/classcv_1_1detail_1_1DpSeamFinder.html#a7914624907986f7a94dd424209a8a609"""  # noqa
        imgs_float = [img.astype(np.float32) for img in imgs]
        return self.finder.find(imgs_float, corners, masks)

    def resize(seam_mask, mask):
        dilated_mask = cv.dilate(seam_mask, None)
        resized_seam_mask = cv.resize(dilated_mask, (mask.shape[1],
                                                     mask.shape[0]),
                                      0, 0, cv.INTER_LINEAR_EXACT)
        return cv.bitwise_and(resized_seam_mask, mask)
