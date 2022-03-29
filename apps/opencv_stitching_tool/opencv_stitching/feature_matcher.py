import math
import cv2 as cv
import numpy as np


class FeatureMatcher:

    MATCHER_CHOICES = ('homography', 'affine')
    DEFAULT_MATCHER = 'homography'
    DEFAULT_RANGE_WIDTH = -1

    def __init__(self,
                 matcher_type=DEFAULT_MATCHER,
                 range_width=DEFAULT_RANGE_WIDTH,
                 **kwargs):

        if matcher_type == "affine":
            """https://docs.opencv.org/4.x/d3/dda/classcv_1_1detail_1_1AffineBestOf2NearestMatcher.html"""  # noqa
            self.matcher = cv.detail_AffineBestOf2NearestMatcher(**kwargs)
        elif range_width == -1:
            """https://docs.opencv.org/4.x/d4/d26/classcv_1_1detail_1_1BestOf2NearestMatcher.html"""  # noqa
            self.matcher = cv.detail_BestOf2NearestMatcher(**kwargs)
        else:
            """https://docs.opencv.org/4.x/d8/d72/classcv_1_1detail_1_1BestOf2NearestRangeMatcher.html"""  # noqa
            self.matcher = cv.detail_BestOf2NearestRangeMatcher(
                range_width, **kwargs
                )

    def match_features(self, features, *args, **kwargs):
        pairwise_matches = self.matcher.apply2(features, *args, **kwargs)
        self.matcher.collectGarbage()
        return pairwise_matches

    @staticmethod
    def draw_matches_matrix(imgs, features, matches, conf_thresh=1,
                            inliers=False, **kwargs):
        matches_matrix = FeatureMatcher.get_matches_matrix(matches)
        for idx1, idx2 in FeatureMatcher.get_all_img_combinations(len(imgs)):
            match = matches_matrix[idx1, idx2]
            if match.confidence < conf_thresh:
                continue
            if inliers:
                kwargs['matchesMask'] = match.getInliers()
            yield idx1, idx2, FeatureMatcher.draw_matches(
                imgs[idx1], features[idx1],
                imgs[idx2], features[idx2],
                match,
                **kwargs
                )

    @staticmethod
    def draw_matches(img1, features1, img2, features2, match1to2, **kwargs):
        kwargs.setdefault('flags', cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        keypoints1 = features1.getKeypoints()
        keypoints2 = features2.getKeypoints()
        matches = match1to2.getMatches()

        return cv.drawMatches(
            img1, keypoints1, img2, keypoints2, matches, None, **kwargs
            )

    @staticmethod
    def get_matches_matrix(pairwise_matches):
        return FeatureMatcher.array_in_sqare_matrix(pairwise_matches)

    @staticmethod
    def get_confidence_matrix(pairwise_matches):
        matches_matrix = FeatureMatcher.get_matches_matrix(pairwise_matches)
        match_confs = [[m.confidence for m in row] for row in matches_matrix]
        match_conf_matrix = np.array(match_confs)
        return match_conf_matrix

    @staticmethod
    def array_in_sqare_matrix(array):
        matrix_dimension = int(math.sqrt(len(array)))
        rows = []
        for i in range(0, len(array), matrix_dimension):
            rows.append(array[i:i+matrix_dimension])
        return np.array(rows)

    def get_all_img_combinations(number_imgs):
        ii, jj = np.triu_indices(number_imgs, k=1)
        for i, j in zip(ii, jj):
            yield i, j

    @staticmethod
    def get_match_conf(match_conf, feature_detector_type):
        if match_conf is None:
            match_conf = \
                FeatureMatcher.get_default_match_conf(feature_detector_type)
        return match_conf

    @staticmethod
    def get_default_match_conf(feature_detector_type):
        if feature_detector_type == 'orb':
            return 0.3
        return 0.65
