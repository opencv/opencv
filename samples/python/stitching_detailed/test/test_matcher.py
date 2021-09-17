import unittest
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..')))

from stitching_detailed.feature_matcher import FeatureMatcher
# %%


class TestMatcher(unittest.TestCase):

    def test_array_in_sqare_matrix(self):
        array = np.zeros(9)
        matrix = FeatureMatcher._array_in_sqare_matrix(array)

        np.testing.assert_array_equal(matrix, np.array([[0., 0., 0.],
                                                        [0., 0., 0.],
                                                        [0., 0., 0.]]))


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
