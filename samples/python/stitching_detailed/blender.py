import cv2 as cv
import numpy as np


class Blender:

    BLENDER_CHOICES = ('multiband', 'feather', 'no',)
    DEFAULT_BLENDER = 'multiband'
    DEFAULT_BLEND_STRENGTH = 5

    def __init__(self, blender_type=DEFAULT_BLENDER,
                 blend_strength=DEFAULT_BLEND_STRENGTH):
        self.blender_type = blender_type
        self.blend_strength = blend_strength
        self.blender = None

    def prepare(self, corners, sizes):
        dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
        blend_width = (np.sqrt(dst_sz[2] * dst_sz[3]) *
                       self.blend_strength / 100)

        if self.blender_type == 'no' or blend_width < 1:
            self.blender = cv.detail.Blender_createDefault(
                cv.detail.Blender_NO
                )

        elif self.blender_type == "multiband":
            self.blender = cv.detail_MultiBandBlender()
            self.blender.setNumBands((np.log(blend_width) /
                                      np.log(2.) - 1.).astype(np.int))

        elif self.blender_type == "feather":
            self.blender = cv.detail_FeatherBlender()
            self.blender.setSharpness(1. / blend_width)

        self.blender.prepare(dst_sz)

    def feed(self, img, mask, corner):
        """https://docs.opencv.org/master/d6/d4a/classcv_1_1detail_1_1Blender.html#a64837308bcf4e414a6219beff6cbe37a"""  # noqa
        self.blender.feed(cv.UMat(img.astype(np.int16)), mask, corner)

    def blend(self):
        """https://docs.opencv.org/master/d6/d4a/classcv_1_1detail_1_1Blender.html#aa0a91ce0d6046d3a63e0123cbb1b5c00"""  # noqa
        result = None
        result_mask = None
        result, result_mask = self.blender.blend(result, result_mask)
        result = cv.convertScaleAbs(result)
        return result
