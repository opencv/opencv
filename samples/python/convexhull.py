#! /usr/bin/env python

print "OpenCV Python version of convexhull"

# import the necessary things for OpenCV
import cv2.cv as cv

# to generate random values
import random

# how many points we want at max
_MAX_POINTS = 100

if __name__ == '__main__':

    # main object to get random values from
    my_random = random.Random ()

    # create the image where we want to display results
    image = cv.CreateImage ( (500, 500), 8, 3)

    # create the window to put the image in
    cv.NamedWindow ('hull', cv.CV_WINDOW_AUTOSIZE)

    while True:
        # do forever

        # get a random number of points
        count = my_random.randrange (0, _MAX_POINTS) + 1

        # initialisations
        points = []

        for i in range (count):
            # generate a random point
            points.append ( (
                my_random.randrange (0, image.width / 2) + image.width / 4,
                my_random.randrange (0, image.width / 2) + image.width / 4
                ))

        # compute the convex hull
        storage = cv.CreateMemStorage(0)
        hull = cv.ConvexHull2 (points, storage, cv.CV_CLOCKWISE, 1)

        # start with an empty image
        cv.SetZero (image)

        # draw all the points as circles in red
        for i in range (count):
            cv.Circle (image, points [i], 2,
                          (0, 0, 255, 0),
                         cv.CV_FILLED, cv.CV_AA, 0)

        # Draw the convex hull as a closed polyline in green
        cv.PolyLine(image, [hull], 1, cv.RGB(0,255,0), 1, cv.CV_AA)

        # display the final image
        cv.ShowImage ('hull', image)

        # handle events, and wait a key pressed
        k = cv.WaitKey (0) % 0x100
        if k == 27:
            # user has press the ESC key, so exit
            break
    cv.DestroyAllWindows()
