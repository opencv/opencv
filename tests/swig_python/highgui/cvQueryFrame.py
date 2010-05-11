#! /usr/bin/env python
"""
This script will test highgui's cvQueryFrame() function
"""

# name of this test and it's requirements
TESTNAME = "cvQueryFrame"
REQUIRED = ["cvGrabFrame","cvRetrieveFrame"]

 
# needed for sys.exit(int) and .works file handling
import os
import sys
import works

# path to imagefiles we need
PREFIX=os.path.join(os.environ["srcdir"],"../../opencv_extra/testdata/python/videos/")

# check requirements and delete old flag file, if it exists
if not works.check_files(REQUIRED,TESTNAME):
	sys.exit(77)

# import the necessary things for OpenCV
from highgui import *
from cv import *


# create a video reader using the tiny video 'uncompressed.avi'
video = cvCreateFileCapture(PREFIX+"uncompressed.avi")

# call cvQueryFrame for 30 frames and check if the returned image is ok
for k in range(0,30):
	image = cvQueryFrame( video )

	if image is None:
	# returned image is not a correct IplImage (pointer),
	# so return an error code
		sys.exit(77)

# ATTENTION: We do not release the video reader, window or any image.
# This is bad manners, but Python and OpenCV don't care...
	
# create flag file for sollowing tests
works.set_file(TESTNAME)

# return 0 ('PASS')
sys.exit(0)
