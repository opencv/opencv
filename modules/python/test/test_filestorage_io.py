#!/usr/bin/env python
"""Algorithm serialization test."""
from __future__ import print_function
import base64
import json
import tempfile
import os
import cv2 as cv
import numpy as np
from tests_common import NewOpenCVTests

class MyData:
    def __init__(self):
        self.A = 97
        self.X = np.pi
        self.name = 'mydata1234'

    def write(self, fs, name):
        fs.startWriteStruct(name, cv.FileNode_MAP|cv.FileNode_FLOW)
        fs.write('A', self.A)
        fs.write('X', self.X)
        fs.write('name', self.name)
        fs.endWriteStruct()

    def read(self, node):
        if (not node.empty()):
            self.A = int(node.getNode('A').real())
            self.X = node.getNode('X').real()
            self.name = node.getNode('name').string()
        else:
            self.A = self.X = 0
            self.name = ''

class filestorage_io_test(NewOpenCVTests):
    strings_data = ['image1.jpg', 'Awesomeness', '../data/baboon.jpg']
    R0 = np.eye(3,3)
    T0 = np.zeros((3,1))

    def write_data(self, fname):
        fs = cv.FileStorage(fname, cv.FileStorage_WRITE)
        R = self.R0
        T = self.T0
        m = MyData()

        fs.write('iterationNr', 100)

        fs.startWriteStruct('strings', cv.FileNode_SEQ)
        for elem in self.strings_data:
            fs.write('', elem)
        fs.endWriteStruct()

        fs.startWriteStruct('Mapping', cv.FileNode_MAP)
        fs.write('One', 1)
        fs.write('Two', 2)
        fs.endWriteStruct()

        fs.write('R_MAT', R)
        fs.write('T_MAT', T)

        m.write(fs, 'MyData')
        fs.release()

    def read_data_and_check(self, fname):
        fs = cv.FileStorage(fname, cv.FileStorage_READ)

        n = fs.getNode('iterationNr')
        itNr = int(n.real())
        self.assertEqual(itNr, 100)

        n = fs.getNode('strings')
        self.assertTrue(n.isSeq())
        self.assertEqual(n.size(), len(self.strings_data))

        for i in range(n.size()):
            self.assertEqual(n.at(i).string(), self.strings_data[i])

        n = fs.getNode('Mapping')
        self.assertEqual(int(n.getNode('Two').real()), 2)
        self.assertEqual(int(n.getNode('One').real()), 1)

        R = fs.getNode('R_MAT').mat()
        T = fs.getNode('T_MAT').mat()

        self.assertEqual(cv.norm(R, self.R0, cv.NORM_INF), 0)
        self.assertEqual(cv.norm(T, self.T0, cv.NORM_INF), 0)

        m0 = MyData()
        m = MyData()
        m.read(fs.getNode('MyData'))
        self.assertEqual(m.A, m0.A)
        self.assertEqual(m.X, m0.X)
        self.assertEqual(m.name, m0.name)

        n = fs.getNode('NonExisting')
        self.assertTrue(n.isNone())
        fs.release()

    def run_fs_test(self, ext):
        fd, fname = tempfile.mkstemp(prefix="opencv_python_sample_filestorage", suffix=ext)
        os.close(fd)
        self.write_data(fname)
        self.read_data_and_check(fname)
        os.remove(fname)

    def test_xml(self):
        self.run_fs_test(".xml")

    def test_yml(self):
        self.run_fs_test(".yml")

    def test_json(self):
        self.run_fs_test(".json")

    def test_base64(self):
        fd, fname = tempfile.mkstemp(prefix="opencv_python_sample_filestorage_base64", suffix=".json")
        os.close(fd)
        np.random.seed(42)
        self.write_base64_json(fname)
        os.remove(fname)

    @staticmethod
    def get_normal_2d_mat(dtype):
        rows = 10
        cols = 20
        cn = 3

        image = np.zeros((rows, cols, cn), dtype)
        image[:] = (1, 2, 127)

        for i in range(rows):
            for j in range(cols):
                image[i, j, 1] = (i + j) % 256

        return image

    @staticmethod
    def get_normal_nd_mat():
        shape = (2, 2, 1, 2)
        cn = 4

        image = np.zeros(shape + (cn,), np.float64)
        image[:] = (0.888, 0.111, 0.666, 0.444)

        return image

    @staticmethod
    def get_empty_2d_mat():
        shape = (0, 0)
        cn = 1

        image = np.zeros(shape + (cn,), np.uint8)

        return image

    @staticmethod
    def get_random_mat():
        rows = 8
        cols = 16
        cn = 1

        image = np.random.rand(rows, cols, cn)

        return image

    @staticmethod
    def decode(data):
        # strip $base64$
        encoded = data[8:]

        if len(encoded) == 0:
            return b''

        # strip info about datatype and padding
        return base64.b64decode(encoded)[24:]

    def write_base64_json(self, fname):
        fs = cv.FileStorage(fname, cv.FileStorage_WRITE_BASE64)

        mats = {'normal_2d_mat_uint8': self.get_normal_2d_mat(np.uint8),
                'normal_2d_mat_int64': self.get_normal_2d_mat(np.int64),
                'normal_2d_mat_uint64': self.get_normal_2d_mat(np.uint64),
                'normal_nd_mat': self.get_normal_nd_mat(),
                'empty_2d_mat': self.get_empty_2d_mat(),
                'random_mat': self.get_random_mat()}

        for name, mat in mats.items():
            fs.write(name, mat)

        fs.release()

        data = {}
        with open(fname) as file:
            data = json.load(file)

        for name, mat in mats.items():
            buffer = b''

            if mat.size != 0:
                if hasattr(mat, 'tobytes'):
                    buffer = mat.tobytes()
                else:
                    buffer = mat.tostring()

            self.assertEqual(buffer, self.decode(data[name]['data']))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
