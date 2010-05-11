#! /usr/bin/env python
"""
This script will test highgui's trackbar functionality
"""

# name if this test and it's requirements
TESTNAME = "cvCreateTrackbar"
REQUIRED = ["cvShowImage"]

# needed for sys.exit(int) and .works file handling
import os
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
bar_name = "brightness"
bar_count= 100


# position of imagefiles we need
PREFIX=os.path.join(os.environ["srcdir"],"../../opencv_extra/testdata/python/images/")

# 'moved' indicates if trackbar has been moved
moved = False

# 'range' indicates if trackbar was outside range [0..bar_count]
range = False


# function to call on a trackbar event
def trackcall( p ):
	# Trackbar position must be in [0..bar_count]
	if (p > bar_count or p < 0):
		globals()["range"] = True

	cvConvertScale( image, image2,float(p)/float(bar_count) )
	cvShowImage( win_name, image2 );
	globals()["moved"] = True


# create output window
cvNamedWindow(win_name,CV_WINDOW_AUTOSIZE)
image  = cvLoadImage(PREFIX+"cvCreateTrackbar.jpg")
image2 = cvLoadImage(PREFIX+"cvCreateTrackbar.jpg")
cvShowImage(win_name,image)

# create the trackbar and save return value
res = cvCreateTrackbar( bar_name, win_name, 0, bar_count, trackcall )

# check return value
if res == 0:
	# something went wrong, so return an error code
	print "(ERROR) Couldn't create trackbar."
	sys.exit(1)
	
# init. window with image
trackcall(bar_count/2)
# reset 'moved' indicator
moved = False

# now give the user 20 seconds to do some input
print "(INFO) Please move trackbar within the next 20 SECONDS to 'PASS' this test."
print "(HINT) You can complete this test prematurely by pressing any key."
print "(INFO) In the case of no user input, the test will be remarked as 'FAIL'."

key = cvWaitKey(20000)

if range:
	# trackbar position value was outside allowed range [0..bar_count]
	print "(ERROR) Trackbar position was outside range."
	sys.exit(1)

if not moved and (key==-1):
	# trackbar has not been moved
	print "(ERROR) No user input detected."
	sys.exit(1)
elif not moved and (key>0):
	# 20sec. passed, trackbar has been moved
	print "(INFO) No trackbar movement detected (but key pressed)."
	sys.exit(77)

# create flag file for following tests
works.set_file(TESTNAME)

# return 0 ('PASS')
sys.exit(0)
