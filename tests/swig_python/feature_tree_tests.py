#!/usr/bin/env python

# 2009-01-16, Xavier Delacour <xavier.delacour@gmail.com>

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

class feature_tree_test(unittest.TestCase):

    def test_kdtree_basic(self):
        n = 1000;
        d = 64;
        query_points = random.rand(n,d)*2-1;
        data = planted_neighbors(query_points)

        tr = cvCreateKDTree(data);
        indices,dist = cvFindFeatures(tr, query_points, 1, 100);

        correct = sum([i == j for j,i in enumerate(indices)])
        assert(correct >= n * .75);

    def test_spilltree_basic(self):
        n = 1000;
        d = 64;
        query_points = random.rand(n,d)*2-1;
        data = planted_neighbors(query_points)

        tr = cvCreateSpillTree(data);
        indices,dist = cvFindFeatures(tr, query_points, 1, 100);

        correct = sum([i == j for j,i in enumerate(indices)])
        assert(correct >= n * .75);

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(feature_tree_test)

if __name__ == '__main__':
    suite = suite()
    unittest.TextTestRunner(verbosity=2).run(suite)

