from collections import OrderedDict
import cv2 as cv
import numpy as np

from .stitching_error import StitchingError


class CameraEstimator:

    CAMERA_ESTIMATOR_CHOICES = OrderedDict()
    CAMERA_ESTIMATOR_CHOICES['homography'] = cv.detail_HomographyBasedEstimator
    CAMERA_ESTIMATOR_CHOICES['affine'] = cv.detail_AffineBasedEstimator

    DEFAULT_CAMERA_ESTIMATOR = list(CAMERA_ESTIMATOR_CHOICES.keys())[0]

    def __init__(self, estimator=DEFAULT_CAMERA_ESTIMATOR, **kwargs):
        self.estimator = CameraEstimator.CAMERA_ESTIMATOR_CHOICES[estimator](
            **kwargs
            )

    def estimate(self, features, pairwise_matches):
        b, cameras = self.estimator.apply(features, pairwise_matches, None)
        if not b:
            raise StitchingError("Homography estimation failed.")
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)
        return cameras
