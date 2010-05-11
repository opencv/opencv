#! /usr/bin/env python
"""
This script will test highgui's mouse functionality
"""

# name of this test and it's requirements
TESTNAME = "cvSetMouseCallback"
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

# global variable which stores information about the pressed mousebuttons
mouse_events = [False,False,False,False,False,False,False,False,False,False]
event_detected = False

# some definitions
win_name = "testing..."
EVENTS = ['CV_EVENT_MOUSEMOVE', 'CV_EVENT_LBUTTONDOWN', 'CV_EVENT_RBUTTONDOWN',  'CV_EVENT_MBUTTONDOWN',  'CV_EVENT_LBUTTONUP',
	  'CV_EVENT_RBUTTONUP', 'CV_EVENT_MBUTTONUP'  , 'CV_EVENT_LBUTTONDBLCLK','CV_EVENT_RBUTTONDBLCLK','CV_EVENT_MBUTTONDBLCLK']


# our callback function, 5th parameter not used here.
def callback_function(event,x,y,flag,param):
	globals()["event_detected"] = True
	# check if event already occured; if not, output info about new event.
	if globals()["mouse_events"][event] == False:
		print "Event "+globals()["EVENTS"][event]+" detected."
		globals()["mouse_events"][event] = True
	return


# create a window ('cvNamedWindow.works' exists, so it must work)
cvNamedWindow(win_name,CV_WINDOW_AUTOSIZE)
# show the baboon in the window
PREFIX = os.path.join(os.environ["srcdir"],"../../opencv_extra/testdata/python/images/")
cvShowImage(win_name, cvLoadImage(PREFIX+"cvSetMouseCallback.jpg"))
# assign callback function 'callback_function' to window, no parameters used here
cvSetMouseCallback( win_name, callback_function )

# give the user information about the test and wait for input
print "(INFO) Please hover the mouse over the baboon image and press"
print "(INFO) your available mousebuttons inside the window to 'PASS' this test."
print "(INFO) You may also perform double-clicks."
print "(INFO) Press a key on your keyboard ot wait 20 seconds to continue."
print "(HINT) If no mouseevent was detected this test will be remarked as 'FAIL'."

# now wait 20 seconds for user to press a key
cvWaitKey(20000)

# reset mouse callback
cvSetMouseCallback( win_name, 0 )
# destroy the window
cvDestroyWindow( win_name )

# check if a mouse event had beed detected
if not event_detected:
	# user didn't interact properly or mouse functionality doesn't work correctly
	print "(ERROR) No mouse event detected."
	sys.exit(1)

# create flag file for following tests
works.set_file(TESTNAME)

# return 0 (success)
sys.exit(0)
