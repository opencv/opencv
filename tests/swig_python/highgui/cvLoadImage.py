"""
This script will test highgui's image loading functionality
for a given parameter of a file extension.
"""


# needed for sys.exit(int) and .works file handling
import os
import sys
import works
from works import *

#import the necessary things for OpenCV
from highgui import *
from cv import *


# some defines
TESTNAME = "cvLoadImage"
REQUIRED = []

# path to imagefiles we need
PREFIX=os.path.join(os.environ["srcdir"],"../../opencv_extra/testdata/python/images/baboon_256x256")


# this functions tries to open an imagefile
# using the filename PREFIX.EXTENSION  and returns True/False 
# on success/fail.

def image_ok( EXTENSION ):
	
	# check requirements and delete old .works file
	WORKSNAME = TESTNAME+'.'+EXTENSION

	if not works.check_files( REQUIRED, WORKSNAME ):
		print "worksfile "+WORKSNAME+" not found."
		return False
	
	image = cvLoadImage(PREFIX+'.'+EXTENSION)

	if image is None:
		return False
	else:
		works.set_file( TESTNAME+EXTENSION )
		return True
