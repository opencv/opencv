from stitching_detailed.feature_detector import FeatureDetector
from stitching_detailed.feature_matcher import FeatureMatcher
from stitching_detailed.subsetter import Subsetter
from stitching_detailed.camera_estimator import CameraEstimator
from stitching_detailed.camera_adjuster import CameraAdjuster
from stitching_detailed.camera_wave_corrector import WaveCorrector


class ImageRegistration:
    def __init__(self,
                 finder=FeatureDetector(),
                 matcher=FeatureMatcher(),
                 subsetter=Subsetter(),
                 camera_estimator=CameraEstimator(),
                 camera_adjuster=CameraAdjuster(),
                 wave_corrector=WaveCorrector()):

        self.finder = finder
        self.matcher = matcher
        self.subsetter = subsetter
        self.camera_estimator = camera_estimator
        self.camera_adjuster = camera_adjuster
        self.wave_corrector = wave_corrector

    def register(self, img_names, images):
        features = self.find_features(images)
        matches = self.match_features(features)
        indices, features, matches = self.subset(
            img_names, features, matches
            )
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.adjust_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        return indices, cameras

    def find_features(self, images):
        return [self.finder.detect_features(img) for img in images]

    def match_features(self, features):
        return self.matcher.match_features(features)

    def subset(self, img_names, features, matches):
        return self.subsetter.subset(img_names, features, matches)

    def estimate_camera_parameters(self, features, matches):
        return self.camera_estimator.estimate(features, matches)

    def adjust_camera_parameters(self, features, matches, cameras):
        return self.camera_adjuster.adjust(features, matches, cameras)

    def perform_wave_correction(self, cameras):
        return self.wave_corrector.correct(cameras)
