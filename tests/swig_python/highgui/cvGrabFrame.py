#! /usr/bin/env python
"""
This script will test highgui's video reading functionality
"""

# name of this test and it's requirements
TESTNAME = "cvGrabFrame"
REQUIRED = ["cvCreateFileCaptureRGBA"]

 
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

# call cvGrabFrame to grab a frame and save return value
res = cvGrabFrame(video)

if res==0:
	print "(ERROR) Couldn't call cvGrabFrame()."
	sys.exit(1)

# ATTENTION: We do not release the video reader, window or any image.
# This is bad manners, but Python and OpenCV don't care...
	
# create flag file for sollowing tests
works.set_file(TESTNAME)

# return 0 ('PASS')
sys.exit(0)
