#! /usr/bin/env python
"""
This script will test highgui's window handle/name functionality
"""

# name of this test and it's requirements
TESTNAME = "cvGetWindowHandleName"
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


# some definitions
win_name = "testing..."

# create a window ( 'cvNamedWindow.works' says: "Ok, go for it!" )
cvNamedWindow(win_name,CV_WINDOW_AUTOSIZE)

# check if the window handle and the according name are correct
win_name_2 = cvGetWindowName( cvGetWindowHandle(win_name) )
if win_name_2!=win_name:
#	print "(ERROR) Incorrect window handle/name."
	sys.exit(1)

# destroy the window
cvDestroyWindow( win_name )

# create flag file for following tests
works.set_file(TESTNAME)

# return 0 ('PASS')
sys.exit(0)
