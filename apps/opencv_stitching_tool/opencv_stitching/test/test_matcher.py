import unittest
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..')))

from opencv_stitching.feature_matcher import FeatureMatcher
# %%


class TestMatcher(unittest.TestCase):

    def test_array_in_sqare_matrix(self):
        array = np.zeros(9)

        matrix = FeatureMatcher.array_in_sqare_matrix(array)

        np.testing.assert_array_equal(matrix, np.array([[0., 0., 0.],
                                                        [0., 0., 0.],
                                                        [0., 0., 0.]]))

    def test_get_all_img_combinations(self):
        nimgs = 3

        combinations = list(FeatureMatcher.get_all_img_combinations(nimgs))

        self.assertEqual(combinations, [(0, 1), (0, 2), (1, 2)])

    def test_get_match_conf(self):
        explicit_match_conf = FeatureMatcher.get_match_conf(1, 'orb')
        implicit_match_conf_orb = FeatureMatcher.get_match_conf(None, 'orb')
        implicit_match_conf_other = FeatureMatcher.get_match_conf(None, 'surf')

        self.assertEqual(explicit_match_conf, 1)
        self.assertEqual(implicit_match_conf_orb, 0.3)
        self.assertEqual(implicit_match_conf_other, 0.65)


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
