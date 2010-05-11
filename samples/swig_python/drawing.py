#! /usr/bin/env python

print "OpenCV Python version of drawing"

# import the necessary things for OpenCV
from opencv import cv
from opencv import highgui

# for making random numbers
from random import Random

def random_color (random):
    """
    Return a random color
    """
    icolor = random.randint (0, 0xFFFFFF)
    return cv.cvScalar (icolor & 0xff, (icolor >> 8) & 0xff, (icolor >> 16) & 0xff)

if __name__ == '__main__':

    # some "constants"
    width = 1000
    height = 700
    window_name = "Drawing Demo"
    number = 100
    delay = 5
    line_type = cv.CV_AA  # change it to 8 to see non-antialiased graphics
    
    # create the source image
    image = cv.cvCreateImage (cv.cvSize (width, height), 8, 3)

    # create window and display the original picture in it
    highgui.cvNamedWindow (window_name, 1)
    cv.cvSetZero (image)
    highgui.cvShowImage (window_name, image)

    # create the random number
    random = Random ()

    # draw some lines
    for i in range (number):
        pt1 = cv.cvPoint (random.randrange (-width, 2 * width),
                          random.randrange (-height, 2 * height))
        pt2 = cv.cvPoint (random.randrange (-width, 2 * width),
                          random.randrange (-height, 2 * height))
        cv.cvLine (image, pt1, pt2,
                   random_color (random),
                   random.randrange (0, 10),
                   line_type, 0)
        
        highgui.cvShowImage (window_name, image)
        highgui.cvWaitKey (delay)

    # draw some rectangles
    for i in range (number):
        pt1 = cv.cvPoint (random.randrange (-width, 2 * width),
                          random.randrange (-height, 2 * height))
        pt2 = cv.cvPoint (random.randrange (-width, 2 * width),
                          random.randrange (-height, 2 * height))
        cv.cvRectangle (image, pt1, pt2,
                        random_color (random),
                        random.randrange (-1, 9),
                        line_type, 0)
        
        highgui.cvShowImage (window_name, image)
        highgui.cvWaitKey (delay)

    # draw some ellipes
    for i in range (number):
        pt1 = cv.cvPoint (random.randrange (-width, 2 * width),
                          random.randrange (-height, 2 * height))
        sz = cv.cvSize (random.randrange (0, 200),
                        random.randrange (0, 200))
        angle = random.randrange (0, 1000) * 0.180
        cv.cvEllipse (image, pt1, sz, angle, angle - 100, angle + 200,
                        random_color (random),
                        random.randrange (-1, 9),
                        line_type, 0)
        
        highgui.cvShowImage (window_name, image)
        highgui.cvWaitKey (delay)

    # init the list of polylines
    nb_polylines = 2
    polylines_size = 3
    pt = [0,] * nb_polylines
    for a in range (nb_polylines):
        pt [a] = [0,] * polylines_size

    # draw some polylines
    for i in range (number):
        for a in range (nb_polylines):
            for b in range (polylines_size):
                pt [a][b] = cv.cvPoint (random.randrange (-width, 2 * width),
                                     random.randrange (-height, 2 * height))
        cv.cvPolyLine (image, pt, 1,
                       random_color (random),
                       random.randrange (1, 9),
                       line_type, 0)

        highgui.cvShowImage (window_name, image)
        highgui.cvWaitKey (delay)

    # draw some filled polylines
    for i in range (number):
        for a in range (nb_polylines):
            for b in range (polylines_size):
                pt [a][b] = cv.cvPoint (random.randrange (-width, 2 * width),
                                     random.randrange (-height, 2 * height))
        cv.cvFillPoly (image, pt,
                       random_color (random),
                       line_type, 0)

        highgui.cvShowImage (window_name, image)
        highgui.cvWaitKey (delay)

    # draw some circles
    for i in range (number):
        pt1 = cv.cvPoint (random.randrange (-width, 2 * width),
                          random.randrange (-height, 2 * height))
        cv.cvCircle (image, pt1, random.randrange (0, 300),
                     random_color (random),
                     random.randrange (-1, 9),
                     line_type, 0)
        
        highgui.cvShowImage (window_name, image)
        highgui.cvWaitKey (delay)

    # draw some text
    for i in range (number):
        pt1 = cv.cvPoint (random.randrange (-width, 2 * width),
                          random.randrange (-height, 2 * height))
        font = cv.cvInitFont (random.randrange (0, 8),
                              random.randrange (0, 100) * 0.05 + 0.01,
                              random.randrange (0, 100) * 0.05 + 0.01,
                              random.randrange (0, 5) * 0.1,
                              random.randrange (0, 10),
                              line_type)

        cv.cvPutText (image, "Testing text rendering!",
                      pt1, font,
                      random_color (random))
        
        highgui.cvShowImage (window_name, image)
        highgui.cvWaitKey (delay)

    # prepare a text, and get it's properties
    font = cv.cvInitFont (cv.CV_FONT_HERSHEY_COMPLEX,
                          3, 3, 0.0, 5, line_type)
    text_size, ymin = cv.cvGetTextSize ("OpenCV forever!", font)
    pt1.x = (width - text_size.width) / 2
    pt1.y = (height + text_size.height) / 2
    image2 = cv.cvCloneImage(image)

    # now, draw some OpenCV pub ;-)
    for i in range (255):
        cv.cvSubS (image2, cv.cvScalarAll (i), image, None)
        cv.cvPutText (image, "OpenCV forever!",
                      pt1, font, cv.cvScalar (255, i, i))
        highgui.cvShowImage (window_name, image)
        highgui.cvWaitKey (delay)

    # wait some key to end
    highgui.cvWaitKey (0)
