#!/usr/bin/env python
import cv2 as cv

from tests_common import NewOpenCVTests

class knearest_test(NewOpenCVTests):
    def test_load(self):
        k_nearest = cv.ml.KNearest_load(self.find_file("ml/opencv_ml_knn.xml"))
        self.assertFalse(k_nearest.empty())
        self.assertTrue(k_nearest.isTrained())

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
