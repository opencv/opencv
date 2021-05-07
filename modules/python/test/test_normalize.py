import numpy as np
from tests_common import NewOpenCVTests
import cv2 as cv

class normalize_test(NewOpenCVTests):

    def test_normalize(self):
        img = np.ones((2, 2))
        img[0, 0] = 10
        # we should be able to call normalize with no additional args now
        res = cv.normalize(img, 0, 1, cv.NORM_MINMAX)
        self.assertEqual(res.tolist(), [[1.0, 0.0], [0.0, 0.0]])
