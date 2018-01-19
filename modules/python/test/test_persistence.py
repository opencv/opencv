#!/usr/bin/env python
""""Core serializaion tests."""
import tempfile
import os
import cv2 as cv
import numpy as np
from tests_common import NewOpenCVTests


class persistence_test(NewOpenCVTests):
    def test_yml_rw(self):
        fd, fname = tempfile.mkstemp(prefix="opencv_python_persistence_", suffix=".yml")
        os.close(fd)

        # Writing ...
        expected = np.array([[[0, 1, 2, 3, 4]]])
        fs = cv.FileStorage(fname, cv.FILE_STORAGE_WRITE)
        fs.write("test", expected)
        fs.release()

        # Reading ...
        fs = cv.FileStorage(fname, cv.FILE_STORAGE_READ)
        root = fs.getFirstTopLevelNode()
        self.assertEqual(root.name(), "test")
        test = fs.getNode("test")
        self.assertEqual(test.empty(), False)
        self.assertEqual(test.name(), "test")
        self.assertEqual(test.type(), cv.FILE_NODE_MAP)
        self.assertEqual(test.isMap(), True)
        actual = test.mat()
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(np.array_equal(expected, actual), True)
        fs.release()

        os.remove(fname)
