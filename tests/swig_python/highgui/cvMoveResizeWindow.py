#! /usr/bin/env python
"""
This script will test highgui's window move/resize functionality
"""

# name of this test and it's requirements
TESTNAME = "cvMoveResizeWindow"
REQUIRED = ["cvNamedWindow"]

 
# needed for sys.exit(int) and .works file handling
import sys
import works

# check requirements and delete old flag file, if it exists
if not works.check_files(REQUIRED,TESTNAME):
	sys.exit(77)

# import the necessary things for OpenCV
from highgui import *
from cv import *

# create a window
cvNamedWindow(TESTNAME, CV_WINDOW_AUTOSIZE)

# move the window around
cvMoveWindow(TESTNAME,   0,   0)
cvMoveWindow(TESTNAME, 100,   0)
cvMoveWindow(TESTNAME, 100, 100)
cvMoveWindow(TESTNAME,   0, 100)

# resize the window
for i in range(1,10):
	cvResizeWindow(TESTNAME, i*100, i*100)

# destroy the window
cvDestroyWindow(TESTNAME)

# create flag file for following tests
works.set_file(TESTNAME)

# return 0 (success)
sys.exit(0)
