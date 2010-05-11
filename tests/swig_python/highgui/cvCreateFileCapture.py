"""
This script will test highgui's video reading functionality
for a given parameter RAW formats.
"""

# needed for sys.exit(int) and .works file handling
import os
import sys
import works
from works import *

# import the necessary things for OpenCV
from highgui import *
from cv import *


# some defines
TESTNAME = "cvCreateFileCapture"
REQUIRED = []
PREFIX   = os.path.join(os.environ["srcdir"],"../../opencv_extra/testdata/python/videos/qcif_")
EXTENSION= ".avi"


# this functions tries to open a videofile
# using the filename PREFIX+FORMAT+.EXTENSION  and returns True/False 
# on success/fail.

def video_ok( FORMAT ):
	
	# check requirements and delete old .works file
	if not works.check_files( REQUIRED, TESTNAME+FORMAT ):
		return false

	filename = PREFIX+FORMAT+EXTENSION
	video = cvCreateFileCapture(PREFIX+FORMAT+EXTENSION)
	if video is None:
		sys.exit(1)
	works.set_file( TESTNAME+FORMAT )
	return True
