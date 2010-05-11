#! /usr/bin/env python
"""
This script will test highgui's cvWaitKey(int) function
"""

# name of this test and it's requirements
TESTNAME = "cvWaitKey"
REQUIRED = ["cvShowImage"]

# needed for sys.exit(int) and .works file handling
import os
import sys
import works

# path to imagefiles we need
PREFIX=os.path.join(os.environ["srcdir"],"../../opencv_extra/testdata/python/images/")

# check requirements and delete old flag file, if it exists
if not works.check_files(REQUIRED, TESTNAME):
	sys.exit(77)


# import the necessary things for OpenCV
from highgui import *

# request some user input
print "(INFO) Press anykey within the next 20 seconds to 'PASS' this test." 

# create a dummy window which reacts on cvWaitKey()
cvNamedWindow(TESTNAME, CV_WINDOW_AUTOSIZE)

# display an image
cvShowImage(TESTNAME, cvLoadImage(PREFIX+"cvWaitKey.jpg"))

# wait 20 seconds using cvWaitKey(20000),
# return 'FAIL' if no key has been pressed.
if cvWaitKey(20000) == -1:
	print "(ERROR) No key pressed, remarking as 'FAIL'."
	sys.exit(1)

#create flag file for the following tests
works.set_file(TESTNAME)

# return 0 ('PASS')
sys.exit(0)
