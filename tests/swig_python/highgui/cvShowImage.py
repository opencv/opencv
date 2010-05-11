#! /usr/bin/env python
"""
This script will test highgui's window functionality
"""

# name of this test and it's requirements
TESTNAME = "cvShowImage"
REQUIRED = ["cvLoadImagejpg", "cvNamedWindow"]

 
# needed for sys.exit(int) and .works file handling
import os
import sys
import works

# path to imagefiles we need
PREFIX=os.path.join(os.environ["srcdir"],"../../opencv_extra/testdata/python/images/")

# check requirements and delete old flag file, if it exists
if not works.check_files(REQUIRED,TESTNAME):
	sys.exit(77)


# import the necessary things for OpenCV
from highgui import *
from cv import *

# defined window name
win_name = "testing..."

# we expect a window to be createable, thanks to 'cvNamedWindow.works'
cvNamedWindow(win_name, CV_WINDOW_AUTOSIZE)

# we expect the image to be loadable, thanks to 'cvLoadImage.works'
image = cvLoadImage(PREFIX+"cvShowImage.jpg")

if image is None:
	print "(ERROR) Couldn't load image "+PREFIX+"cvShowImage.jpg"
	sys.exit(1)

# try to show image in window
res = cvShowImage( win_name, image )
cvWaitKey(0)


if res == 0:
	cvReleaseImage(image)
	cvDestroyWindow(win_name)
	sys.exit(1)
	
# destroy window
cvDestroyWindow(win_name)

# create flag file for following tests
works.set_file(TESTNAME)

# return 0 ('PASS')
sys.exit(0)
	
	
	
