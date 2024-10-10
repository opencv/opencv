#!/usr/bin/env python
"""Algorithm serialization test."""
import tempfile
import os
import cv2 as cv
from tests_common import NewOpenCVTests


class algorithm_rw_test(NewOpenCVTests):
    def test_algorithm_rw(self):
        fd, fname = tempfile.mkstemp(prefix="opencv_python_algorithm_", suffix=".yml")
        os.close(fd)

        # some arbitrary non-default parameters
        gold = cv.ORB_create(nfeatures=200, scaleFactor=1.3, nlevels=5, edgeThreshold=28)
        gold.write(cv.FileStorage(fname, cv.FILE_STORAGE_WRITE), "ORB")

        fs = cv.FileStorage(fname, cv.FILE_STORAGE_READ)
        algorithm = cv.ORB_create()
        algorithm.read(fs.getNode("ORB"))

        self.assertEqual(algorithm.getMaxFeatures(), 200)
        self.assertEqual(algorithm.getScaleFactor(), 1.3)
        self.assertEqual(algorithm.getNLevels(), 5)
        self.assertEqual(algorithm.getEdgeThreshold(), 28)

        os.remove(fname)
