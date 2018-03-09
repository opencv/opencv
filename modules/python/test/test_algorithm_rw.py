#!/usr/bin/env python
"""Algorithm serializaion test."""
import tempfile
import os
import cv2 as cv
from tests_common import NewOpenCVTests


class algorithm_rw_test(NewOpenCVTests):
    def test_algorithm_rw(self):
        fd, fname = tempfile.mkstemp(prefix="opencv_python_algorithm_", suffix=".yml")
        os.close(fd)

        # some arbitrary non-default parameters
        gold = cv.AKAZE_create(descriptor_size=1, descriptor_channels=2, nOctaves=3, threshold=4.0)
        gold.write(cv.FileStorage(fname, cv.FILE_STORAGE_WRITE), "AKAZE")

        fs = cv.FileStorage(fname, cv.FILE_STORAGE_READ)
        algorithm = cv.AKAZE_create()
        algorithm.read(fs.getNode("AKAZE"))

        self.assertEqual(algorithm.getDescriptorSize(), 1)
        self.assertEqual(algorithm.getDescriptorChannels(), 2)
        self.assertEqual(algorithm.getNOctaves(), 3)
        self.assertEqual(algorithm.getThreshold(), 4.0)

        os.remove(fname)
