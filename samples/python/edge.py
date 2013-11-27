#! /usr/bin/env python

print "OpenCV Python version of edge"

import sys
import urllib2
import cv2.cv as cv

# some definitions
win_name = "Edge"
trackbar_name = "Threshold"

# the callback on the trackbar
def on_trackbar(position):

    cv.Smooth(gray, edge, cv.CV_BLUR, 3, 3, 0)
    cv.Not(gray, edge)

    # run the edge dector on gray scale
    cv.Canny(gray, edge, position, position * 3, 3)

    # reset
    cv.SetZero(col_edge)

    # copy edge points
    cv.Copy(im, col_edge, edge)

    # show the im
    cv.ShowImage(win_name, col_edge)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        im = cv.LoadImage( sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
    else:
        url = 'https://raw.github.com/Itseez/opencv/master/samples/c/fruits.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        im = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_COLOR)

    # create the output im
    col_edge = cv.CreateImage((im.width, im.height), 8, 3)

    # convert to grayscale
    gray = cv.CreateImage((im.width, im.height), 8, 1)
    edge = cv.CreateImage((im.width, im.height), 8, 1)
    cv.CvtColor(im, gray, cv.CV_BGR2GRAY)

    # create the window
    cv.NamedWindow(win_name, cv.CV_WINDOW_AUTOSIZE)

    # create the trackbar
    cv.CreateTrackbar(trackbar_name, win_name, 1, 100, on_trackbar)

    # show the im
    on_trackbar(0)

    # wait a key pressed to end
    cv.WaitKey(0)
    cv.DestroyAllWindows()
