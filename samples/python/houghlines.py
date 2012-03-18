#!/usr/bin/python
# This is a standalone program. Pass an image name as a first parameter of the program.

import sys
from math import sin, cos, sqrt, pi
import cv2.cv as cv
import urllib2

# toggle between CV_HOUGH_STANDARD and CV_HOUGH_PROBILISTIC
USE_STANDARD = True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        src = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        url = 'http://code.opencv.org/svn/opencv/trunk/opencv/doc/pics/building.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        src = cv.DecodeImageM(imagefiledata, cv.CV_LOAD_IMAGE_GRAYSCALE)


    cv.NamedWindow("Source", 1)
    cv.NamedWindow("Hough", 1)

    while True:
        dst = cv.CreateImage(cv.GetSize(src), 8, 1)
        color_dst = cv.CreateImage(cv.GetSize(src), 8, 3)
        storage = cv.CreateMemStorage(0)
        lines = 0
        cv.Canny(src, dst, 50, 200, 3)
        cv.CvtColor(dst, color_dst, cv.CV_GRAY2BGR)

        if USE_STANDARD:
            lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_STANDARD, 1, pi / 180, 100, 0, 0)
            for (rho, theta) in lines[:100]:
                a = cos(theta)
                b = sin(theta)
                x0 = a * rho 
                y0 = b * rho
                pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
                pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
                cv.Line(color_dst, pt1, pt2, cv.RGB(255, 0, 0), 3, 8)
        else:
            lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_PROBABILISTIC, 1, pi / 180, 50, 50, 10)
            for line in lines:
                cv.Line(color_dst, line[0], line[1], cv.CV_RGB(255, 0, 0), 3, 8)

        cv.ShowImage("Source", src)
        cv.ShowImage("Hough", color_dst)

        k = cv.WaitKey(0) % 0x100
        if k == ord(' '):
            USE_STANDARD = not USE_STANDARD
        if k == 27:
            break
    cv.DestroyAllWindows()
