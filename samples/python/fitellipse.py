#!/usr/bin/python
"""
This program is a demonstration of ellipse fitting.

Trackbar controls threshold parameter.

Gray lines are contours.  Colored lines are fit ellipses.

Original C implementation by:  Denis Burenkov.
Python implementation by: Roman Stanchak, James Bowman
"""

import sys
import urllib2
import random
import cv2.cv as cv

def contour_iterator(contour):
    while contour:
        yield contour
        contour = contour.h_next()

class FitEllipse:

    def __init__(self, source_image, slider_pos):
        self.source_image = source_image
        cv.CreateTrackbar("Threshold", "Result", slider_pos, 255, self.process_image)
        self.process_image(slider_pos)

    def process_image(self, slider_pos): 
        """
        This function finds contours, draws them and their approximation by ellipses.
        """
        stor = cv.CreateMemStorage()
        
        # Create the destination images
        image02 = cv.CloneImage(self.source_image)
        cv.Zero(image02)
        image04 = cv.CreateImage(cv.GetSize(self.source_image), cv.IPL_DEPTH_8U, 3)
        cv.Zero(image04)

        # Threshold the source image. This needful for cv.FindContours().
        cv.Threshold(self.source_image, image02, slider_pos, 255, cv.CV_THRESH_BINARY)

        # Find all contours.
        cont = cv.FindContours(image02,
            stor,
            cv.CV_RETR_LIST,
            cv.CV_CHAIN_APPROX_NONE,
            (0, 0))

        for c in contour_iterator(cont):
            # Number of points must be more than or equal to 6 for cv.FitEllipse2
            if len(c) >= 6:
                # Copy the contour into an array of (x,y)s
                PointArray2D32f = cv.CreateMat(1, len(c), cv.CV_32FC2)
                for (i, (x, y)) in enumerate(c):
                    PointArray2D32f[0, i] = (x, y)
                
                # Draw the current contour in gray
                gray = cv.CV_RGB(100, 100, 100)
                cv.DrawContours(image04, c, gray, gray,0,1,8,(0,0))
                
                # Fits ellipse to current contour.
                (center, size, angle) = cv.FitEllipse2(PointArray2D32f)
                
                # Convert ellipse data from float to integer representation.
                center = (cv.Round(center[0]), cv.Round(center[1]))
                size = (cv.Round(size[0] * 0.5), cv.Round(size[1] * 0.5))
                
                # Draw ellipse in random color
                color = cv.CV_RGB(random.randrange(256),random.randrange(256),random.randrange(256))
                cv.Ellipse(image04, center, size,
                          angle, 0, 360,
                          color, 2, cv.CV_AA, 0)

        # Show image. HighGUI use.
        cv.ShowImage( "Result", image04 )


if __name__ == '__main__':
    if len(sys.argv) > 1:
        source_image = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        url = 'https://code.ros.org/svn/opencv/trunk/opencv/samples/c/stuff.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        source_image = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_GRAYSCALE)
    
    # Create windows.
    cv.NamedWindow("Source", 1)
    cv.NamedWindow("Result", 1)

    # Show the image.
    cv.ShowImage("Source", source_image)

    fe = FitEllipse(source_image, 70)

    print "Press any key to exit"
    cv.WaitKey(0)

    cv.DestroyWindow("Source")
    cv.DestroyWindow("Result")
