#!/usr/bin/python
import urllib2
import sys
import cv2.cv as cv
import numpy

# SRGB-linear conversions using NumPy - see http://en.wikipedia.org/wiki/SRGB

def srgb2lin(x):
    a = 0.055
    return numpy.where(x <= 0.04045,
                       x * (1.0 / 12.92),
                       numpy.power((x + a) * (1.0 / (1 + a)), 2.4))

def lin2srgb(x):
    a = 0.055
    return numpy.where(x <= 0.0031308,
                       x * 12.92,
                       (1 + a) * numpy.power(x, 1 / 2.4) - a)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img0 = cv.LoadImageM( sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
    else:
        url = 'http://code.opencv.org/svn/opencv/trunk/opencv/samples/c/lena.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        img0 = cv.DecodeImageM(imagefiledata, cv.CV_LOAD_IMAGE_COLOR)

    cv.NamedWindow("original", 1)
    cv.ShowImage("original", img0)

    # Image was originally bytes in range 0-255.  Turn it into an array of floats in range 0.0 - 1.0
    n = numpy.asarray(img0) / 255.0

    # Use NumPy to do some transformations on the image

    # Negate the image by subtracting it from 1.0
    cv.NamedWindow("negative")
    cv.ShowImage("negative", cv.fromarray(1.0 - n))

    # Assume the image was sRGB, and compute the linear version.
    cv.NamedWindow("linear")
    cv.ShowImage("linear", cv.fromarray(srgb2lin(n)))

    # Look at a subwindow
    cv.NamedWindow("subwindow")
    cv.ShowImage("subwindow", cv.fromarray(n[200:300,200:400]))

    # Compute the grayscale image
    cv.NamedWindow("monochrome")
    ln = srgb2lin(n)
    red = ln[:,:,0]
    grn = ln[:,:,1]
    blu = ln[:,:,2]
    linear_mono = 0.3 * red + 0.59 * grn + 0.11 * blu
    cv.ShowImage("monochrome", cv.fromarray(lin2srgb(linear_mono)))

    # Apply a blur to the NumPy array using OpenCV
    cv.NamedWindow("gaussian")
    cv.Smooth(n, n, cv.CV_GAUSSIAN, 15, 15)
    cv.ShowImage("gaussian", cv.fromarray(n))

    cv.WaitKey(0)
