import cv2 as cv


class FeatureMatcher:

    choices = ('homography', 'affine')
    default = 'homography'

    def __init__(self,
                 matcher_type=default,
                 range_width=-1,
                 *args, **kwargs):

        if matcher_type == "affine":
            """https://docs.opencv.org/master/d3/dda/classcv_1_1detail_1_1AffineBestOf2NearestMatcher.html"""  # noqa
            self.matcher = cv.detail_AffineBestOf2NearestMatcher(
                *args, **kwargs
                )
        elif range_width == -1:
            """https://docs.opencv.org/master/d4/d26/classcv_1_1detail_1_1BestOf2NearestMatcher.html"""  # noqa
            self.matcher = cv.detail.BestOf2NearestMatcher_create(
                *args, **kwargs
                )
        else:
            """https://docs.opencv.org/master/d8/d72/classcv_1_1detail_1_1BestOf2NearestRangeMatcher.html"""  # noqa
            self.matcher = cv.detail.BestOf2NearestRangeMatcher_create(
                range_width, *args, **kwargs
                )

    def match_features(self, features, *args, **kwargs):
        pairwise_matches = self.matcher.apply2(features, *args, **kwargs)
        self.matcher.collectGarbage()
        return pairwise_matches

    def get_default_match_conf(feature_detector):
        if feature_detector == 'orb':
            return 0.3
        else:
            return 0.65
