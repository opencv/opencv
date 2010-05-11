"""
This script will test highgui's cvQueryFrame() function
for different video formats
"""

# import the necessary things for OpenCV and comparson routine
import os
from highgui import *
from cv import *
import match

# path to videos and images we need
PREFIX=os.path.join(os.environ["srcdir"],"../../opencv_extra/testdata/python/")

# this is the folder with the videos and images
# and name of output window
IMAGES		= PREFIX+"images/"
VIDEOS		= PREFIX+"videos/"

# testing routine, called for each entry in FILENAMES
# and compares each frame with corresponding frame in COMPARISON
def query_ok(FILENAME,ERRORS):

    # create a video reader using the tiny videofile VIDEOS+FILENAME
    video=cvCreateFileCapture(VIDEOS+FILENAME)

    if video is None:
	# couldn't open video (FAIL)
	return 1

    # call cvQueryFrame for 29 frames and check if the returned image is ok
    for k in range(29):
    	image=cvQueryFrame(video)

	if image is None:
	# returned image is NULL (FAIL)
		return 1

	if not match.match(image,k,ERRORS[k]):
		return 1
	
    cvReleaseCapture(video)
    # everything is fine (PASS)
    return 0
