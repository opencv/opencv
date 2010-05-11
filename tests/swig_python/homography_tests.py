#!/usr/bin/env python

# 2009-01-16, Xavier Delacour <xavier.delacour@gmail.com>

import unittest
from numpy import *;
from numpy.linalg import *;
import sys;

import cvtestutils
from cv import *;
from adaptors import *;

def transform(H,x):
    x1 = H * asmatrix(r_[x[0],x[1],1]).transpose()
    x1 = asarray(x1).flatten()
    return r_[x1[0]/x1[2],x1[1]/x1[2]]

class homography_test(unittest.TestCase):

    def test_ransac_identity(self):
        pts1 = random.rand(100,2);
        result,H = cvFindHomography(pts1, pts1, CV_RANSAC, 1e-5);
        assert(result and all(abs(Ipl2NumPy(H) - eye(3)) < 1e-5));

    def test_ransac_0_outliers(self):
        pts1 = random.rand(100,2);
        H1 = asmatrix(random.rand(3,3));
        H1 = H1 / H1[2,2]
        pts2 = [transform(H1,x) for x in pts1]
        result,H = cvFindHomography(pts1, pts2, CV_RANSAC, 1e-5);
        assert(result and all(abs(H1-H)<1e-5))

    def test_ransac_30_outliers(self):
        pts1 = random.rand(100,2);
        H1 = asmatrix(random.rand(3,3));
        H1 = H1 / H1[2,2]
        pts2 = [transform(H1,x) for x in pts1]
        pts2[0:30] = random.rand(30,2)
        result,H = cvFindHomography(pts1, pts2, CV_RANSAC, 1e-5);
        assert(result and all(abs(H1-H)<1e-5))

    def test_ransac_70_outliers(self):
        pts1 = random.rand(100,2);
        H1 = asmatrix(random.rand(3,3));
        H1 = H1 / H1[2,2]
        pts2 = [transform(H1,x) for x in pts1]
        pts2[0:70] = random.rand(70,2)
        result,H = cvFindHomography(pts1, pts2, CV_RANSAC, 1e-5);
        assert(result and all(abs(H1-H)<1e-5))

    def test_ransac_90_outliers(self):
        pts1 = random.rand(100,2);
        H1 = asmatrix(random.rand(3,3));
        H1 = H1 / H1[2,2]
        pts2 = [transform(H1,x) for x in pts1]
        pts2[0:90] = random.rand(90,2)
        result,H = cvFindHomography(pts1, pts2, CV_RANSAC, 1e-5);
        assert(not result or not all(abs(H1-H)<1e-5))

    def test_lmeds_identity(self):
        pts1 = random.rand(100,2);
        result,H = cvFindHomography(pts1, pts1, CV_LMEDS);
        assert(result and all(abs(Ipl2NumPy(H) - eye(3)) < 1e-5));

    def test_lmeds_0_outliers(self):
        pts1 = random.rand(100,2);
        H1 = asmatrix(random.rand(3,3));
        H1 = H1 / H1[2,2]
        pts2 = [transform(H1,x) for x in pts1]
        result,H = cvFindHomography(pts1, pts2, CV_LMEDS);
        assert(result and all(abs(H1-H)<1e-5))

    def test_lmeds_30_outliers(self):
        pts1 = random.rand(100,2);
        H1 = asmatrix(random.rand(3,3));
        H1 = H1 / H1[2,2]
        pts2 = [transform(H1,x) for x in pts1]
        pts2[0:30] = random.rand(30,2)
        result,H = cvFindHomography(pts1, pts2, CV_LMEDS);
        assert(result and all(abs(H1-H)<1e-5))

    def test_lmeds_70_outliers(self):
        pts1 = random.rand(100,2);
        H1 = asmatrix(random.rand(3,3));
        H1 = H1 / H1[2,2]
        pts2 = vstack([transform(H1,x) for x in pts1])
        pts2[0:70] = random.rand(70,2)
        result,H = cvFindHomography(pts1, pts2, CV_LMEDS);
        assert(not result or not all(abs(H1-H)<1e-5))

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(homography_test)

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

