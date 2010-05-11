#!/usr/bin/env python
import unittest
import cvtestutils
from cv import *

class cmp_test(unittest.TestCase):
    def setUp(self):
        self.w=17
        self.h=17
        self.x0 = cvCreateMat(self.w,self.h,CV_32F)
        self.x1 = cvCreateMat(self.w,self.h,CV_32F)
        cvSet(self.x0, cvScalarAll(0.0))
        cvSet(self.x1, cvScalarAll(1.0))
    def check_format(self, y):
        assert( y.rows == self.h )
        assert( y.cols == self.w )
        assert( CV_MAT_DEPTH(y.type)==CV_8U )
    def check_allzero(self, y):
        assert( cvCountNonZero(y)==0 )
    def check_all255(self, y):
        nonzero=cvCountNonZero(y)
        assert( nonzero==self.w*self.h )
        sum = cvSum(y)[0]
        assert( sum == self.w*self.h*255 )

    def test_CvMat_gt(self):
        y=self.x1>0
        self.check_format( y )
        self.check_all255( y )
        y=self.x0>0
        self.check_format( y )
        self.check_allzero( y )

    def test_CvMat_gte(self):
        y=self.x1>=0
        self.check_format( y )
        self.check_all255( y )
        y=self.x0>=0
        self.check_format( y )
        self.check_all255( y )

    def test_CvMat_lt(self):
        y=self.x1<1
        self.check_format( y )
        self.check_allzero( y )
        y=self.x0<1
        self.check_format( y )
        self.check_all255( y )

    def test_CvMat_lte(self):
        y=self.x1<=1
        self.check_format( y )
        self.check_all255( y )
        y=self.x0<=1
        self.check_format( y )

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(cmp_test)

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
