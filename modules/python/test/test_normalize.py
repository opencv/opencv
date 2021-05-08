import numpy as np
from tests_common import NewOpenCVTests
import cv2 as cv

class normalize_test(NewOpenCVTests):

    def test_normalize(self):
        img = np.ones((2, 2))
        # we should be able to call normalize with no additional args now
        res = cv.normalize(img)
        self.assertEqual(res.tolist(), [[0.5, 0.5], [0.5, 0.5]])
