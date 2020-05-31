#!/usr/bin/env python
"""Algorithm serialization test."""
from __future__ import print_function
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

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
