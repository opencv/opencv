from collections import OrderedDict
import cv2 as cv


class FeatureDetector:
    DETECTOR_CHOICES = OrderedDict()
    try:
        cv.xfeatures2d_SURF.create()  # check if the function can be called
        DETECTOR_CHOICES['surf'] = cv.xfeatures2d_SURF.create
    except (AttributeError, cv.error):
        print("SURF not available")

    # if SURF not available, ORB is default
    DETECTOR_CHOICES['orb'] = cv.ORB.create

    try:
        DETECTOR_CHOICES['sift'] = cv.SIFT_create
    except AttributeError:
        print("SIFT not available")

    try:
        DETECTOR_CHOICES['brisk'] = cv.BRISK_create
    except AttributeError:
        print("BRISK not available")

    try:
        DETECTOR_CHOICES['akaze'] = cv.AKAZE_create
    except AttributeError:
        print("AKAZE not available")

    DEFAULT_DETECTOR = list(DETECTOR_CHOICES.keys())[0]

    def __init__(self, detector=DEFAULT_DETECTOR, **kwargs):
        self.detector = FeatureDetector.DETECTOR_CHOICES[detector](**kwargs)

    def detect_features(self, img, *args, **kwargs):
        return cv.detail.computeImageFeatures2(self.detector, img,
                                               *args, **kwargs)

    @staticmethod
    def draw_keypoints(img, features, **kwargs):
        kwargs.setdefault('color', (0, 255, 0))
        keypoints = features.getKeypoints()
        return cv.drawKeypoints(img, keypoints, None, **kwargs)
