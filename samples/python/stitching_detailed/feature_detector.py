from collections import OrderedDict
import cv2 as cv


class FeatureDetector:
    choices = OrderedDict()
    try:
        cv.xfeatures2d_SURF.create()  # check if the function can be called
        choices['surf'] = cv.xfeatures2d_SURF.create
    except (AttributeError, cv.error):
        print("SURF not available")
    choices['orb'] = cv.ORB.create    # if SURF not available, ORB is default
    try:
        choices['sift'] = cv.xfeatures2d_SIFT.create
    except AttributeError:
        print("SIFT not available")
    try:
        choices['brisk'] = cv.BRISK_create
    except AttributeError:
        print("BRISK not available")
    try:
        choices['akaze'] = cv.AKAZE_create
    except AttributeError:
        print("AKAZE not available")

    default = list(choices.keys())[0]

    def __init__(self, detector=default, *args, **kwargs):
        self.detector = FeatureDetector.choices[detector](*args, **kwargs)

    def detect_features(self, img, *args, **kwargs):
        return cv.detail.computeImageFeatures2(self.detector, img,
                                               *args, **kwargs)
