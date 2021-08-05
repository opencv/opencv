from stitching_detailed.feature_detector import FeatureDetector
from stitching_detailed.feature_matcher import FeatureMatcher
from stitching_detailed.subsetter import Subsetter
from stitching_detailed.camera_estimator import CameraEstimator
from stitching_detailed.camera_adjuster import CameraAdjuster
from stitching_detailed.camera_wave_corrector import WaveCorrector

class ImageRegistration:
    def __init__(self,
                 finder=FeatureDetector.DEFAULT_DETECTOR,
                 matcher=FeatureMatcher.DEFAULT_MATCHER,
                 matcher_rangewidth=FeatureMatcher.DEFAULT_RANGE_WIDTH,
                 matcher_try_cuda=False,
                 matcher_match_conf=None,
                 conf_thresh=Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
                 estimator=CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
                 ba=CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
                 ba_refine_mask=CameraAdjuster.DEFAULT_REFINEMENT_MASK,
                 wave_correct=WaveCorrector.DEFAULT_WAVE_CORRECTION):

        self.finder = FeatureDetector(finder)

        if not matcher_match_conf:
            matcher_match_conf = FeatureMatcher.get_default_match_conf(finder)

        self.matcher = FeatureMatcher(matcher, matcher_rangewidth,
                                      try_use_gpu=matcher_try_cuda,
                                      match_conf=matcher_match_conf)
        self.subsetter = Subsetter(conf_thresh)
        self.camera_estimator = CameraEstimator(estimator)
        self.camera_adjuster = CameraAdjuster(ba, ba_refine_mask)
        self.wave_corrector = WaveCorrector(wave_correct)

    def register(self, images):
        features = self.find_features(images)
        matches = self.match_features(features)
        indices, features, matches = self.subset(features, matches)
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.adjust_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        return indices, cameras

    def find_features(self, images):
        return [self.finder.detect_features(img) for img in images]

    def match_features(self, features):
        return self.matcher.match_features(features)

    def subset(self, features, matches):
        subsetter = Subsetter(features, matches)
        return subsetter.subset(features, matches)

    def estimate_camera_parameters(self, features, matches):
        return self.camera_estimator.estimate(features, matches)

    def adjust_camera_parameters(self, features, matches, cameras):
        return self.camera_adjuster.adjust(features, matches, cameras)

    def perform_wave_correction(self, cameras):
        return self.wave_corrector.correct(cameras)
