#!/usr/bin/python
import cv2.cv as cv
import sys
import urllib2

if __name__ == "__main__":
    cv.NamedWindow("win")
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        im = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_GRAYSCALE)
        im3 = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_COLOR)
    else:
        url = 'http://code.opencv.org/svn/opencv/trunk/opencv/samples/cpp/left01.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        im = cv.DecodeImageM(imagefiledata, cv.CV_LOAD_IMAGE_GRAYSCALE)
        im3 = cv.DecodeImageM(imagefiledata, cv.CV_LOAD_IMAGE_COLOR)
    chessboard_dim = ( 9, 6 )
    
    found_all, corners = cv.FindChessboardCorners( im, chessboard_dim )
    print found_all, len(corners)

    cv.DrawChessboardCorners( im3, chessboard_dim, corners, found_all )
    
    cv.ShowImage("win", im3);
    cv.WaitKey()
