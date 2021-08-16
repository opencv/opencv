import statistics

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

    def register(self, img_data):
        features = self.find_features(img_data.get_work_images())
        matches = self.match_features(features)
        img_data, features, matches = self.subset(
            img_data, features, matches
            )
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.adjust_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        scale = self.estimate_scale(cameras)
        return img_data, cameras, scale

    def find_features(self, images):
        return [self.finder.detect_features(img) for img in images]

    def match_features(self, features):
        return self.matcher.match_features(features)

    def subset(self, img_data, features, matches):
        self.subsetter.save_matches_graph_dot_file(img_data.img_names, matches)
        indices = self.subsetter.get_indices_to_keep(features, matches)
        img_data.update_indices(indices)
        features = Subsetter.subset_list(features, indices)
        matches = Subsetter.subset_matches(matches, indices)
        return img_data, features, matches

    def estimate_camera_parameters(self, features, matches):
        return self.camera_estimator.estimate(features, matches)

    def adjust_camera_parameters(self, features, matches, cameras):
        return self.camera_adjuster.adjust(features, matches, cameras)

    def perform_wave_correction(self, cameras):
        return self.wave_corrector.correct(cameras)

    def estimate_scale(self, cameras):
        focals = [cam.focal for cam in cameras]
        return statistics.median(focals)
