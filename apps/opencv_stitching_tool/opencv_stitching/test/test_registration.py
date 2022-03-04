import unittest
import os
import sys

import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..')))

from opencv_stitching.feature_detector import FeatureDetector
from opencv_stitching.feature_matcher import FeatureMatcher
from opencv_stitching.subsetter import Subsetter


class TestImageRegistration(unittest.TestCase):

    def test_feature_detector(self):
        img1 = cv.imread("s1.jpg")

        default_number_of_keypoints = 500
        detector = FeatureDetector("orb")
        features = detector.detect_features(img1)
        self.assertEqual(len(features.getKeypoints()),
                         default_number_of_keypoints)

        other_keypoints = 1000
        detector = FeatureDetector("orb", nfeatures=other_keypoints)
        features = detector.detect_features(img1)
        self.assertEqual(len(features.getKeypoints()), other_keypoints)

    def test_feature_matcher(self):
        img1, img2 = cv.imread("s1.jpg"), cv.imread("s2.jpg")

        detector = FeatureDetector("orb")
        features = [detector.detect_features(img1),
                    detector.detect_features(img2)]

        matcher = FeatureMatcher()
        pairwise_matches = matcher.match_features(features)
        self.assertEqual(len(pairwise_matches), len(features)**2)
        self.assertGreater(pairwise_matches[1].confidence, 2)

        matches_matrix = FeatureMatcher.get_matches_matrix(pairwise_matches)
        self.assertEqual(matches_matrix.shape, (2, 2))
        conf_matrix = FeatureMatcher.get_confidence_matrix(pairwise_matches)
        self.assertTrue(np.array_equal(
            conf_matrix > 2,
            np.array([[False,  True], [True, False]])
            ))

    def test_subsetting(self):
        img1, img2 = cv.imread("s1.jpg"), cv.imread("s2.jpg")
        img3, img4 = cv.imread("boat1.jpg"), cv.imread("boat2.jpg")
        img5 = cv.imread("boat3.jpg")
        img_names = ["s1.jpg", "s2.jpg", "boat1.jpg", "boat2.jpg", "boat3.jpg"]

        detector = FeatureDetector("orb")
        features = [detector.detect_features(img1),
                    detector.detect_features(img2),
                    detector.detect_features(img3),
                    detector.detect_features(img4),
                    detector.detect_features(img5)]
        matcher = FeatureMatcher()
        pairwise_matches = matcher.match_features(features)
        subsetter = Subsetter(confidence_threshold=1,
                              matches_graph_dot_file="dot_graph.txt")  # view in https://dreampuf.github.io  # noqa

        indices = subsetter.get_indices_to_keep(features, pairwise_matches)
        indices_to_delete = subsetter.get_indices_to_delete(len(img_names),
                                                            indices)

        np.testing.assert_array_equal(indices, np.array([2, 3, 4]))
        np.testing.assert_array_equal(indices_to_delete, np.array([0, 1]))

        subsetted_image_names = subsetter.subset_list(img_names, indices)
        self.assertEqual(subsetted_image_names,
                         ['boat1.jpg', 'boat2.jpg', 'boat3.jpg'])

        matches_subset = subsetter.subset_matches(pairwise_matches, indices)
        # FeatureMatcher.get_confidence_matrix(pairwise_matches)
        # FeatureMatcher.get_confidence_matrix(subsetted_matches)
        self.assertEqual(pairwise_matches[13].confidence,
                         matches_subset[1].confidence)

        graph = subsetter.get_matches_graph(img_names, pairwise_matches)
        self.assertTrue(graph.startswith("graph matches_graph{"))

        subsetter.save_matches_graph_dot_file(img_names, pairwise_matches)
        with open('dot_graph.txt', 'r') as file:
            graph = file.read()
            self.assertTrue(graph.startswith("graph matches_graph{"))


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
