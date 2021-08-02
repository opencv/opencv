from collections import OrderedDict
import cv2 as cv
import numpy as np


class CameraEstimator:

    choices = OrderedDict()
    choices['homography'] = cv.detail_HomographyBasedEstimator
    choices['affine'] = cv.detail_AffineBasedEstimator

    default = list(choices.keys())[0]

    def __init__(self, estimator=default, *args, **kwargs):
        self.estimator = CameraEstimator.choices[estimator](*args, **kwargs)

    def estimate(self, features, pairwise_matches):
        b, cameras = self.estimator.apply(features, pairwise_matches, None)
        if not b:
            print("Homography estimation failed.")
            exit()
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)
        return cameras
