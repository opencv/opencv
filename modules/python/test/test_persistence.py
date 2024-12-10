#!/usr/bin/env python
""""Core serialization tests."""
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
        expected = np.array([[[0, 1, 2, 3, 4]]], dtype=np.intc)
        expected_str = ("Hello", "World", "!")
        fs = cv.FileStorage(fname, cv.FILE_STORAGE_WRITE)
        fs.write("test", expected)
        fs.write("strings", expected_str)
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

        strings = fs.getNode("strings")
        self.assertEqual(strings.isSeq(), True)
        self.assertEqual(strings.size(), len(expected_str))
        self.assertEqual(all(strings.at(i).isString() for i in range(strings.size())), True)
        self.assertSequenceEqual([strings.at(i).string() for i in range(strings.size())], expected_str)
        fs.release()

        os.remove(fname)
