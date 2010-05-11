#! /usr/bin/env python

print "OpenCV Python version of contours"

# import the necessary things for OpenCV
from opencv import cv
from opencv import highgui

# some default constants
_SIZE = 500
_DEFAULT_LEVEL = 3

# definition of some colors
_red = cv.cvScalar (0, 0, 255, 0);
_green = cv.cvScalar (0, 255, 0, 0);
_white = cv.cvRealScalar (255)
_black = cv.cvRealScalar (0)

# the callback on the trackbar, to set the level of contours we want
# to display
def on_trackbar (position):

    # create the image for putting in it the founded contours
    contours_image = cv.cvCreateImage (cv.cvSize (_SIZE, _SIZE), 8, 3)

    # compute the real level of display, given the current position
    levels = position - 3

    # initialisation
    _contours = contours
    
    if levels <= 0:
        # zero or negative value
        # => get to the nearest face to make it look more funny
        _contours = contours.h_next.h_next.h_next
        
    # first, clear the image where we will draw contours
    cv.cvSetZero (contours_image)
    
    # draw contours in red and green
    cv.cvDrawContours (contours_image, _contours,
                       _red, _green,
                       levels, 3, cv.CV_AA,
                       cv.cvPoint (0, 0))

    # finally, show the image
    highgui.cvShowImage ("contours", contours_image)

if __name__ == '__main__':

    # create the image where we want to display results
    image = cv.cvCreateImage (cv.cvSize (_SIZE, _SIZE), 8, 1)

    # start with an empty image
    cv.cvSetZero (image)

    # draw the original picture
    for i in range (6):
        dx = (i % 2) * 250 - 30
        dy = (i / 2) * 150
        
        cv.cvEllipse (image,
                      cv.cvPoint (dx + 150, dy + 100),
                      cv.cvSize (100, 70),
                      0, 0, 360, _white, -1, 8, 0)
        cv.cvEllipse (image,
                      cv.cvPoint (dx + 115, dy + 70),
                      cv.cvSize (30, 20),
                      0, 0, 360, _black, -1, 8, 0)
        cv.cvEllipse (image,
                      cv.cvPoint (dx + 185, dy + 70),
                      cv.cvSize (30, 20),
                      0, 0, 360, _black, -1, 8, 0)
        cv.cvEllipse (image,
                      cv.cvPoint (dx + 115, dy + 70),
                      cv.cvSize (15, 15),
                      0, 0, 360, _white, -1, 8, 0)
        cv.cvEllipse (image,
                      cv.cvPoint (dx + 185, dy + 70),
                      cv.cvSize (15, 15),
                      0, 0, 360, _white, -1, 8, 0)
        cv.cvEllipse (image,
                      cv.cvPoint (dx + 115, dy + 70),
                      cv.cvSize (5, 5),
                      0, 0, 360, _black, -1, 8, 0)
        cv.cvEllipse (image,
                      cv.cvPoint (dx + 185, dy + 70),
                      cv.cvSize (5, 5),
                      0, 0, 360, _black, -1, 8, 0)
        cv.cvEllipse (image,
                      cv.cvPoint (dx + 150, dy + 100),
                      cv.cvSize (10, 5),
                      0, 0, 360, _black, -1, 8, 0)
        cv.cvEllipse (image,
                      cv.cvPoint (dx + 150, dy + 150),
                      cv.cvSize (40, 10),
                      0, 0, 360, _black, -1, 8, 0)
        cv.cvEllipse (image,
                      cv.cvPoint (dx + 27, dy + 100),
                      cv.cvSize (20, 35),
                      0, 0, 360, _white, -1, 8, 0)
        cv.cvEllipse (image,
                      cv.cvPoint (dx + 273, dy + 100),
                      cv.cvSize (20, 35),
                      0, 0, 360, _white, -1, 8, 0)

    # create window and display the original picture in it
    highgui.cvNamedWindow ("image", 1)
    highgui.cvShowImage ("image", image)

    # create the storage area
    storage = cv.cvCreateMemStorage (0)
    
    # find the contours
    nb_contours, contours = cv.cvFindContours (image,
                                               storage,
                                               cv.sizeof_CvContour,
                                               cv.CV_RETR_TREE,
                                               cv.CV_CHAIN_APPROX_SIMPLE,
                                               cv.cvPoint (0,0))

    # comment this out if you do not want approximation
    contours = cv.cvApproxPoly (contours, cv.sizeof_CvContour,
                                storage,
                                cv.CV_POLY_APPROX_DP, 3, 1)
    
    # create the window for the contours
    highgui.cvNamedWindow ("contours", 1)

    # create the trackbar, to enable the change of the displayed level
    highgui.cvCreateTrackbar ("levels+3", "contours", 3, 7, on_trackbar)

    # call one time the callback, so we will have the 1st display done
    on_trackbar (_DEFAULT_LEVEL)

    # wait a key pressed to end
    highgui.cvWaitKey (0)
