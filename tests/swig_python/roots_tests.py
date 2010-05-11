#!/usr/bin/env python

# 2009-01-16, Xavier Delacour <xavier.delacour@gmail.com>

import unittest
from numpy import *;
from numpy.linalg import *;
import sys;

import cvtestutils
from cv import *;
from adaptors import *;

## these are mostly to test bindings, since there are more/better tests in tests/cxcore

def verify(x,y):
    x = Ipl2NumPy(x);
    assert(all(abs(x - y)<1e-4));

class roots_test(unittest.TestCase):
    
    def test_cvSolvePoly(self):
        
        verify(cvSolvePoly(asmatrix([-1,1]).astype(float64)),
                    array([[(1.000000, 0.000000)]]));

        verify(cvSolvePoly(asmatrix([-1,1]).astype(float32)),
                    array([[(1.000000, 0.000000)]]));

        verify(cvSolvePoly(asmatrix([-1,0,0,0,0,1]).astype(float64)),
               array([[(1, 0)],[(0.309017, 0.951057)],[(0.309017, -0.951057)],
                      [(-0.809017, 0.587785)],[(-0.809017, -0.587785)]]))

        verify(cvSolvePoly(asmatrix([-1,0,0,0,0,1]).astype(float32)),
               array([[(1, 0)],[(0.309017, 0.951057)],[(0.309017, -0.951057)],
                      [(-0.809017, 0.587785)],[(-0.809017, -0.587785)]]))

    def test_cvSolveCubic(self):

        verify(cvSolveCubic(asmatrix([-1,0,0,1]).astype(float32))[1],
               array([[1],[0],[0]]));

        verify(cvSolveCubic(asmatrix([-1,0,0,1]).astype(float64))[1],
               array([[1],[0],[0]]));

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(roots_test)

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

