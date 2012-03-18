#!/usr/bin/python
import urllib2
import sys
import cv2.cv as cv
import numpy

if __name__ == "__main__":
    cv.NamedWindow("camera", 1)

    capture = cv.CaptureFromCAM(0)

    paste = cv.CreateMat(960, 1280, cv.CV_8UC3)
    topleft = numpy.asarray(cv.GetSubRect(paste, (0, 0, 640, 480)))
    topright = numpy.asarray(cv.GetSubRect(paste, (640, 0, 640, 480)))
    bottomleft = numpy.asarray(cv.GetSubRect(paste, (0, 480, 640, 480)))
    bottomright = numpy.asarray(cv.GetSubRect(paste, (640, 480, 640, 480)))

    while True:
        img = cv.GetMat(cv.QueryFrame(capture))

        n = (numpy.asarray(img)).astype(numpy.uint8)

        red = n[:,:,0]
        grn = n[:,:,1]
        blu = n[:,:,2]

        topleft[:,:,0] = 255 - grn
        topleft[:,:,1] = red
        topleft[:,:,2] = blu

        topright[:,:,0] = blu
        topright[:,:,1] = 255 - red
        topright[:,:,2] = grn

        bottomright[:,:,0] = red
        bottomright[:,:,1] = grn
        bottomright[:,:,2] = 255 - blu

        fgrn = grn.astype(numpy.float32)
        fred = red.astype(numpy.float32)
        bottomleft[:,:,0] = blu
        bottomleft[:,:,1] = (abs(fgrn - fred)).astype(numpy.uint8)
        bottomleft[:,:,2] = red

        cv.ShowImage("camera", paste)
        if cv.WaitKey(6) == 27:
            break
    cv.DestroyAllWindows()
