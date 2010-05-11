#!/usr/bin/env python

"""A simple TestCase class for testing the adaptors.py module.

2007-11-xx, Vicent Mas <vmas@carabos.com> Carabos Coop. V.
2007-11-08, minor modifications for distribution, Mark Asbach <asbach@ient.rwth-aachen.de>
"""

import unittest
import os

import PIL.Image
import numpy

import cvtestutils
import cv
import highgui
import adaptors

import sys

class AdaptorsTestCase(unittest.TestCase):
    def test00_array_interface(self):
        """Check if PIL supports the array interface."""
        self.assert_(PIL.Image.VERSION>='1.1.6',
            """The installed PIL library doesn't support the array """
            """interface. Please, update to version 1.1.6b2 or higher.""")


    def test01_PIL2NumPy(self):
        """Test the adaptors.PIL2NumPy function."""

        a = adaptors.PIL2NumPy(self.pil_image)
        self.assert_(a.flags['WRITEABLE'] == True,
            'PIL2NumPy should return a writeable array.')
        b = numpy.asarray(self.pil_image)
        self.assert_((a == b).all() == True,
            'The returned numpy array has not been properly constructed.')


    def test02_NumPy2PIL(self):
        """Test the adaptors.NumPy2PIL function."""

        a = numpy.asarray(self.pil_image)
        b = adaptors.NumPy2PIL(a)
        self.assert_(self.pil_image.tostring() == b.tostring(),
            'The returned image has not been properly constructed.')


    def test03_Ipl2PIL(self):
        """Test the adaptors.Ipl2PIL function."""
    
        i = adaptors.Ipl2PIL(self.ipl_image)
        self.assert_(self.pil_image.tostring() == i.tostring(),
            'The returned image has not been properly constructed.')


    def test04_PIL2Ipl(self):
        """Test the adaptors.PIL2Ipl function."""

        i = adaptors.PIL2Ipl(self.pil_image)
        self.assert_(self.ipl_image.imageData == i.imageData,
            'The returned image has not been properly constructed.')


    def test05_Ipl2NumPy(self):
        """Test the adaptors.Ipl2NumPy function."""
    
        a = adaptors.Ipl2NumPy(self.ipl_image)
        a_1d = numpy.reshape(a, (a.size, ))
        # For 3-channel IPL images  the order of channels will be BGR
        # but NumPy array order of channels will be RGB so a conversion
        # is needed before we can compare both images
        if self.ipl_image.nChannels == 3:
            rgb = cv.cvCreateImage(cv.cvSize(self.ipl_image.width, self.ipl_image.height), self.ipl_image.depth, 3)
            cv.cvCvtColor(self.ipl_image, rgb, cv.CV_BGR2RGB)
            self.assert_(a_1d.tostring() == rgb.imageData,
                'The returned image has not been properly constructed.')
        else:
            self.assert_(a_1d.tostring() == self.ipl_image.imageData,
                'The returned image has not been properly constructed.')


    def test06_NumPy2Ipl(self):
        """Test the adaptors.NumPy2Ipl function."""

        a = adaptors.Ipl2NumPy(self.ipl_image)
        b = adaptors.NumPy2Ipl(a)
        self.assert_(self.ipl_image.imageData == b.imageData,
            'The returned image has not been properly constructed.')

    def load_image( self, fname ):
        self.ipl_image = highgui.cvLoadImage(fname, 4|2)
        self.pil_image = PIL.Image.open(fname, 'r')

class AdaptorsTestCase1(AdaptorsTestCase):
    def setUp( self ):
        self.load_image( os.path.join(cvtestutils.datadir(),'images','cvSetMouseCallback.jpg'))

class AdaptorsTestCase2(AdaptorsTestCase):
    def setUp( self ):
        self.load_image( os.path.join(cvtestutils.datadir(),'images','baboon.jpg'))

def suite():
    cases=[]
    cases.append( unittest.TestLoader().loadTestsFromTestCase( AdaptorsTestCase1 ) )
    cases.append( unittest.TestLoader().loadTestsFromTestCase( AdaptorsTestCase2 ) )

    return unittest.TestSuite(cases)

if __name__ == '__main__':
        unittest.TextTestRunner(verbosity=2).run(suite())

