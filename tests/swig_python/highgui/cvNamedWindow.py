#! /usr/bin/env python
"""
This script will test highgui's window functionality
"""

# name of this test and it's requirements
TESTNAME = "cvNamedWindow"
REQUIRED = []

# needed for sys.exit(int) and .works file handling
import sys
import works

# check requirements and delete flag file if it exists
if not works.check_files( REQUIRED, TESTNAME ):
	sys.exit(77)


# import the necessary things for OpenCV
from highgui import *
from cv import *

# some definitions
win_name = "testing..."

# create a window and save return code
res = cvNamedWindow(win_name,CV_WINDOW_AUTOSIZE)

# if returncode is ok, window creation was sucessful
if res == 0:
	# something went wrong, so return an errorcode
	sys.exit(1)

# destroy the window
cvDestroyWindow( win_name )

# create flag file for following tests
works.set_file(TESTNAME)

# return 0 ('PASS')
sys.exit(0)
