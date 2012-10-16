#! /usr/bin/env python

print "OpenCV Python version of contours"

# import the necessary things for OpenCV
import cv2.cv as cv

# some default constants
_SIZE = 500
_DEFAULT_LEVEL = 3

# definition of some colors
_red =  (0, 0, 255, 0);
_green =  (0, 255, 0, 0);
_white = cv.RealScalar (255)
_black = cv.RealScalar (0)

# the callback on the trackbar, to set the level of contours we want
# to display
def on_trackbar (position):

    # create the image for putting in it the founded contours
    contours_image = cv.CreateImage ( (_SIZE, _SIZE), 8, 3)

    # compute the real level of display, given the current position
    levels = position - 3

    # initialisation
    _contours = contours

    if levels <= 0:
        # zero or negative value
        # => get to the nearest face to make it look more funny
        _contours = contours.h_next().h_next().h_next()

    # first, clear the image where we will draw contours
    cv.SetZero (contours_image)

    # draw contours in red and green
    cv.DrawContours (contours_image, _contours,
                       _red, _green,
                       levels, 3, cv.CV_AA,
                        (0, 0))

    # finally, show the image
    cv.ShowImage ("contours", contours_image)

if __name__ == '__main__':

    # create the image where we want to display results
    image = cv.CreateImage ( (_SIZE, _SIZE), 8, 1)

    # start with an empty image
    cv.SetZero (image)

    # draw the original picture
    for i in range (6):
        dx = (i % 2) * 250 - 30
        dy = (i / 2) * 150

        cv.Ellipse (image,
                       (dx + 150, dy + 100),
                       (100, 70),
                      0, 0, 360, _white, -1, 8, 0)
        cv.Ellipse (image,
                       (dx + 115, dy + 70),
                       (30, 20),
                      0, 0, 360, _black, -1, 8, 0)
        cv.Ellipse (image,
                       (dx + 185, dy + 70),
                       (30, 20),
                      0, 0, 360, _black, -1, 8, 0)
        cv.Ellipse (image,
                       (dx + 115, dy + 70),
                       (15, 15),
                      0, 0, 360, _white, -1, 8, 0)
        cv.Ellipse (image,
                       (dx + 185, dy + 70),
                       (15, 15),
                      0, 0, 360, _white, -1, 8, 0)
        cv.Ellipse (image,
                       (dx + 115, dy + 70),
                       (5, 5),
                      0, 0, 360, _black, -1, 8, 0)
        cv.Ellipse (image,
                       (dx + 185, dy + 70),
                       (5, 5),
                      0, 0, 360, _black, -1, 8, 0)
        cv.Ellipse (image,
                       (dx + 150, dy + 100),
                       (10, 5),
                      0, 0, 360, _black, -1, 8, 0)
        cv.Ellipse (image,
                       (dx + 150, dy + 150),
                       (40, 10),
                      0, 0, 360, _black, -1, 8, 0)
        cv.Ellipse (image,
                       (dx + 27, dy + 100),
                       (20, 35),
                      0, 0, 360, _white, -1, 8, 0)
        cv.Ellipse (image,
                       (dx + 273, dy + 100),
                       (20, 35),
                      0, 0, 360, _white, -1, 8, 0)

    # create window and display the original picture in it
    cv.NamedWindow ("image", 1)
    cv.ShowImage ("image", image)

    # create the storage area
    storage = cv.CreateMemStorage (0)

    # find the contours
    contours = cv.FindContours(image,
                               storage,
                               cv.CV_RETR_TREE,
                               cv.CV_CHAIN_APPROX_SIMPLE,
                               (0,0))

    # comment this out if you do not want approximation
    contours = cv.ApproxPoly (contours,
                                storage,
                                cv.CV_POLY_APPROX_DP, 3, 1)

    # create the window for the contours
    cv.NamedWindow ("contours", 1)

    # create the trackbar, to enable the change of the displayed level
    cv.CreateTrackbar ("levels+3", "contours", 3, 7, on_trackbar)

    # call one time the callback, so we will have the 1st display done
    on_trackbar (_DEFAULT_LEVEL)

    # wait a key pressed to end
    cv.WaitKey (0)
    cv.DestroyAllWindows()
