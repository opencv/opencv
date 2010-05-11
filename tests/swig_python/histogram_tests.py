#!/usr/bin/env python
import unittest
import sys
import os
import cvtestutils
from cv import cvCreateHist, cvGetSize, cvCreateImage, cvCvtColor, cvSplit, cvCalcHist, cvCalcArrHist, CV_HIST_ARRAY
from highgui import cvLoadImage

image_fname = os.path.join( cvtestutils.datadir(), 'images', 'baboon_256x256.bmp' )
class HistogramTestCase( unittest.TestCase ):
    def setUp(self):
        frame = cvLoadImage( image_fname )
        frame_size = cvGetSize( frame )
        r = cvCreateImage (frame_size, 8, 1)
        g = cvCreateImage (frame_size, 8, 1)
        b = cvCreateImage (frame_size, 8, 1)

        cvSplit( frame, r, g, b, None)
        self.rgb = (r,g,b)
        assert(frame is not None)

        hist_size = [64, 64, 64]
        ranges = [ [0, 255], [0, 255], [0, 255] ]
        self.hist = cvCreateHist( hist_size, CV_HIST_ARRAY, ranges, 1 )

    def test_cvCreateHist( self ):
        assert( self.hist is not None )

    def test_cvCalcArrHist(self):
        cvCalcArrHist( self.rgb, self.hist, 0, None)

    def test_cvCalcHist(self):
        cvCalcHist( self.rgb, self.hist, 0, None)

def suite():
    tests = ['test_cvCreateHist', 'test_cvCalcArrHist', 'test_cvCalcHist']
    return unittest.TestSuite( map(HistogramTestCase, tests))

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
