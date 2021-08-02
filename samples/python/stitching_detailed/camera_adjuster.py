from collections import OrderedDict
import cv2 as cv
import numpy as np


class CameraAdjuster:
    """https://docs.opencv.org/master/d5/d56/classcv_1_1detail_1_1BundleAdjusterBase.html"""  # noqa

    choices = OrderedDict()
    choices['ray'] = cv.detail_BundleAdjusterRay
    choices['reproj'] = cv.detail_BundleAdjusterReproj
    choices['affine'] = cv.detail_BundleAdjusterAffinePartial
    choices['no'] = cv.detail_NoBundleAdjuster

    default = list(choices.keys())[0]

    def __init__(self, adjuster=default):
        self.adjuster = CameraAdjuster.choices[adjuster]()
        self.adjuster.setConfThresh(1)

    def set_refinement_mask(self, bundle_adjustment_refine_mask):
        refine_mask = np.zeros((3, 3), np.uint8)
        if bundle_adjustment_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if bundle_adjustment_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if bundle_adjustment_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if bundle_adjustment_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if bundle_adjustment_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        self.adjuster.setRefinementMask(refine_mask)

    def adjust(self, features, pairwise_matches, estimated_cameras):
        b, cameras = self.adjuster.apply(features,
                                         pairwise_matches,
                                         estimated_cameras)
        if not b:
            print("Camera parameters adjusting failed.")
            exit()

        return cameras
