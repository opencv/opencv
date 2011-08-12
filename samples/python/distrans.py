#!/usr/bin/python
import sys
import cv2.cv as cv
import urllib2

wndname = "Distance transform"
tbarname = "Threshold"

# The output images
dist = 0
dist8u1 = 0
dist8u2 = 0
dist8u = 0
dist32s = 0

gray = 0
edge = 0

# define a trackbar callback
def on_trackbar(edge_thresh):

    cv.Threshold(gray, edge, float(edge_thresh), float(edge_thresh), cv.CV_THRESH_BINARY)
    #Distance transform                  
    cv.DistTransform(edge, dist, cv.CV_DIST_L2, cv.CV_DIST_MASK_5)

    cv.ConvertScale(dist, dist, 5000.0, 0)
    cv.Pow(dist, dist, 0.5)
    
    cv.ConvertScale(dist, dist32s, 1.0, 0.5)
    cv.AndS(dist32s, cv.ScalarAll(255), dist32s, None)
    cv.ConvertScale(dist32s, dist8u1, 1, 0)
    cv.ConvertScale(dist32s, dist32s, -1, 0)
    cv.AddS(dist32s, cv.ScalarAll(255), dist32s, None)
    cv.ConvertScale(dist32s, dist8u2, 1, 0)
    cv.Merge(dist8u1, dist8u2, dist8u2, None, dist8u)
    cv.ShowImage(wndname, dist8u)


if __name__ == "__main__":
    edge_thresh = 100

    if len(sys.argv) > 1:
        gray = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        url = 'https://code.ros.org/svn/opencv/trunk/opencv/samples/c/stuff.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        gray = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_GRAYSCALE)

    # Create the output image
    dist = cv.CreateImage((gray.width, gray.height), cv.IPL_DEPTH_32F, 1)
    dist8u1 = cv.CloneImage(gray)
    dist8u2 = cv.CloneImage(gray)
    dist8u = cv.CreateImage((gray.width, gray.height), cv.IPL_DEPTH_8U, 3)
    dist32s = cv.CreateImage((gray.width, gray.height), cv.IPL_DEPTH_32S, 1)

    # Convert to grayscale
    edge = cv.CloneImage(gray)

    # Create a window
    cv.NamedWindow(wndname, 1)

    # create a toolbar 
    cv.CreateTrackbar(tbarname, wndname, edge_thresh, 255, on_trackbar)

    # Show the image
    on_trackbar(edge_thresh)

    # Wait for a key stroke; the same function arranges events processing
    cv.WaitKey(0)
