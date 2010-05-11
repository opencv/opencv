#! /usr/bin/env python

print "OpenCV Python version of convexhull"

# import the necessary things for OpenCV
from opencv import cv
from opencv import highgui

# to generate random values
import random

# how many points we want at max
_MAX_POINTS = 100

if __name__ == '__main__':

    # main object to get random values from
    my_random = random.Random ()

    # create the image where we want to display results
    image = cv.cvCreateImage (cv.cvSize (500, 500), 8, 3)

    # create the window to put the image in
    highgui.cvNamedWindow ('hull', highgui.CV_WINDOW_AUTOSIZE)

    while True:
        # do forever

        # get a random number of points
        count = my_random.randrange (0, _MAX_POINTS) + 1

        # initialisations
        points = []
        
        for i in range (count):
            # generate a random point
            points.append (cv.cvPoint (
                my_random.randrange (0, image.width / 2) + image.width / 4,
                my_random.randrange (0, image.width / 2) + image.width / 4
                ))

        # compute the convex hull
        hull = cv.cvConvexHull2 (points, cv.CV_CLOCKWISE, 0)

        # start with an empty image
        cv.cvSetZero (image)

        for i in range (count):
            # draw all the points
            cv.cvCircle (image, points [i], 2,
                         cv.cvScalar (0, 0, 255, 0),
                         cv.CV_FILLED, cv.CV_AA, 0)

        # start the line from the last point
        pt0 = points [hull [-1]]
        
        for point_index in hull:
            # connect the previous point to the current one

            # get the current one
            pt1 = points [point_index]

            # draw
            cv.cvLine (image, pt0, pt1,
                       cv.cvScalar (0, 255, 0, 0),
                       1, cv.CV_AA, 0)

            # now, current one will be the previous one for the next iteration
            pt0 = pt1

        # display the final image
        highgui.cvShowImage ('hull', image)

        # handle events, and wait a key pressed
        k = highgui.cvWaitKey (0)
        if k == '\x1b':
            # user has press the ESC key, so exit
            break
