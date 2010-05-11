"""
This script will compare tho images and decides with a threshold
if these to images are "equal enough"
"""

# import the necessary things for OpenCV
from cv import *
from highgui import *

import frames
import sys
import os

PREFIX=os.path.join(os.environ["srcdir"],"../../opencv_extra/testdata/python/images/")


DisplayImages=False

if DisplayImages:
	videowindow="video"
	referencewindow="reference"
	cvNamedWindow(videowindow,CV_WINDOW_AUTOSIZE)
	cvNamedWindow(referencewindow,CV_WINDOW_AUTOSIZE)

# returns True/False if match/non-match
def match( image, index, thres ):

# load image from comparison set
	QCIFcompare=cvLoadImage(PREFIX+frames.QCIF[index])

	if QCIFcompare is None:
		print "Couldn't open image "+PREFIX+frames.QCIF[index]+" for comparison!"
		sys.exit(1)

# resize comparison image to input image dimensions
	size=cvSize(image.width,image.height)
	compare=cvCreateImage(size,IPL_DEPTH_8U,image.nChannels)
	cvResize(QCIFcompare,compare)

# compare images
	diff=cvNorm( image, compare, CV_RELATIVE_L2 )

	if DisplayImages:
		cvShowImage(videowindow,image)
		cvShowImage(referencewindow,compare)
		if diff<=thres:
			cvWaitKey(200)
		else:
			print "index==",index,": max==",thres," is==",diff
			cvWaitKey(5000)

	cvReleaseImage(QCIFcompare)
	cvReleaseImage(compare)

	if diff<=thres:
		return True
	else:
		return False


