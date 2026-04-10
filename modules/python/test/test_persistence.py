#!/usr/bin/env python
""""Core serialization tests."""
import tempfile
import os
import cv2 as cv
import numpy as np
import yaml
from tests_common import NewOpenCVTests


class persistence_test(NewOpenCVTests):
    def test_yml_rw(self):
        fd, fname = tempfile.mkstemp(prefix="opencv_python_persistence_", suffix=".yml")
        os.close(fd)

        # Writing ...
        expected = np.array([[[0, 1, 2, 3, 4]]])
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

    def test_yml_python_interop(self):
            ref_data = {
                'int_value': 42,
                "bool_value": True,
                "float_value": 3.1415926,
                "int64_value": 2147483647 + 1024, # C++ INT_MAX + 1024
                "string_value": "opencv"
            }

            fd, test_file_name = tempfile.mkstemp(prefix="opencv_python_persistence_", suffix=".yml")
            os.close(fd)

            with open(test_file_name, 'w') as ff:
                yaml.dump(ref_data, ff)

            # Notice: no cv.FileStorage_FORMAT_YAML flag needed now thanks to the C++ fix!
            fs = cv.FileStorage(test_file_name, cv.FILE_STORAGE_READ)
            self.assertTrue(fs.isOpened())

            node = fs.getNode('int_value')
            self.assertTrue(node.isInt())
            self.assertEqual(42, int(node.real()))

            node = fs.getNode('int64_value')
            self.assertTrue(node.isInt())
            self.assertEqual(2147483647 + 1024, int(node.real()))

            node = fs.getNode('float_value')
            self.assertTrue(node.isReal())
            self.assertEqual(3.1415926, node.real())

            node = fs.getNode('string_value')
            self.assertTrue(node.isString())
            self.assertEqual("opencv", node.string())

            fs.release()
            os.remove(test_file_name)