#!/usr/bin/env python
import cvtestutils
import unittest
from cv import *

class moments_test(unittest.TestCase):
    def setUp(self):
        # create an image
        img = cvCreateMat(100,100,CV_8U);

        cvZero( img )
        # draw a rectangle in the middle
        cvRectangle( img, cvPoint( 25, 25 ), cvPoint( 75, 75 ), CV_RGB(255,255,255), -1 );
        
        self.img = img

        # create the storage area
        self.storage = cvCreateMemStorage (0)

        # find the contours
        nb_contours, self.contours = cvFindContours (img,
            self.storage,
            sizeof_CvContour,
            CV_RETR_LIST,
            CV_CHAIN_APPROX_SIMPLE,
            cvPoint (0,0))

    def test_cvMoments_CvMat( self ):
        m = CvMoments()
        cvMoments( self.img, m, 1 )
    def test_cvMoments_CvSeq( self ):
        m = CvMoments()
        # Now test with CvSeq
        for contour in self.contours.hrange():
            cvMoments( contour, m, 1 )

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(moments_test)

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
