#!/usr/bin/env python

# 2009-01-12, Xavier Delacour <xavier.delacour@gmail.com>

# gdb --cd ~/opencv-lsh/tests/python --args /usr/bin/python lsh_tests.py
# set env PYTHONPATH /home/x/opencv-lsh/debug/interfaces/swig/python:/home/x/opencv-lsh/debug/lib
# export PYTHONPATH=/home/x/opencv-lsh/debug/interfaces/swig/python:/home/x/opencv-lsh/debug/lib

import unittest
from numpy import *;
from numpy.linalg import *;
import sys;

import cvtestutils
from cv import *;
from adaptors import *;

def planted_neighbors(query_points, R = .4):
    n,d = query_points.shape
    data = zeros(query_points.shape)
    for i in range(0,n):
        a = random.rand(d)
        a = random.rand()*R*a/sqrt(sum(a**2))
        data[i] = query_points[i] + a
    return data

class lsh_test(unittest.TestCase):

    def test_basic(self):
        n = 10000;
        d = 64;
        query_points = random.rand(n,d)*2-1;
        data = planted_neighbors(query_points)

        lsh = cvCreateMemoryLSH(d, n);
        cvLSHAdd(lsh, data);
        indices,dist = cvLSHQuery(lsh, query_points, 1, 100);
        correct = sum([i == j for j,i in enumerate(indices)])
        assert(correct >= n * .75);

    def test_sensitivity(self):
        n = 10000;
        d = 64;
        query_points = random.rand(n,d);
        data = random.rand(n,d);

        lsh = cvCreateMemoryLSH(d, 1000, 10, 10);
        cvLSHAdd(lsh, data);

        good = 0
        trials = 20
        print 
        for x in query_points[0:trials]:
            x1 = asmatrix(x) # PyArray_to_CvArr doesn't like 1-dim arrays
            indices,dist = cvLSHQuery(lsh, x1, n, n);
            indices = Ipl2NumPy(indices)
            indices = unique(indices[where(indices>=0)])

            brute = vstack([(sqrt(sum((a-x)**2)),i,0) for i,a in enumerate(data)])
            lshp = vstack([(sqrt(sum((x-data[i])**2)),i,1) for i in indices])
            combined = vstack((brute,lshp))
            combined = combined[argsort(combined[:,0])]

            spread = [i for i,a in enumerate(combined[:,2]) if a==1]
            spread = histogram(spread,bins=4,new=True)[0]
            print spread, sum(diff(spread)<0)
            if sum(diff(spread)<0) == 3: good = good + 1
        print good,"pass"
        assert(good > trials * .75);

    def test_remove(self):
        n = 10000;
        d = 64;
        query_points = random.rand(n,d)*2-1;
        data = planted_neighbors(query_points)
        lsh = cvCreateMemoryLSH(d, n);
        indices = cvLSHAdd(lsh, data);
        assert(LSHSize(lsh)==n);
        cvLSHRemove(lsh,indices[0:n/2])
        assert(LSHSize(lsh)==n/2);

    def test_destroy(self):
        n = 10000;
        d = 64;
        lsh = cvCreateMemoryLSH(d, n);

    def test_destroy2(self):
        n = 10000;
        d = 64;
        query_points = random.rand(n,d)*2-1;
        data = planted_neighbors(query_points)
        lsh = cvCreateMemoryLSH(d, n);
        indices = cvLSHAdd(lsh, data);


# move this to another file

# img1 = cvLoadImage(img1_fn);
# img2 = cvLoadImage(img2_fn);
# pts1,desc1 = cvExtractSURF(img1); # * make util routine to extract points and descriptors
# pts2,desc2 = cvExtractSURF(img2);
# lsh = cvCreateMemoryLSH(d, n);
# cvLSHAdd(lsh, desc1);
# indices,dist = cvLSHQuery(lsh, desc2, 2, 100);
# matches = [((pts1[x[0]].pt.x,pts1[x[0]].pt.y),(pts2[j].pt.x,pts2[j].pt.y)) \
#            for j,x in enumerate(hstack((indices,dist))) \
#            if x[2] and (not x[3] or x[2]/x[3]>.6)]
# out = cvCloneImage(img1);
# for p1,p2 in matches:
#     cvCircle(out,p1,3,CV_RGB(255,0,0));
#     cvLine(out,p1,p2,CV_RGB(100,100,100));
# cvNamedWindow("matches");
# cvShowImage("matches",out);
# cvWaitKey(0);

        
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(lsh_test)

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

