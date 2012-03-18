#! /usr/bin/env python

import cv2.cv as cv

cap = cv.CreateFileCapture("../c/tree.avi")
img = cv.QueryFrame(cap)
print "Got frame of dimensions (", img.width, " x ", img.height, ")"

cv.NamedWindow("win", cv.CV_WINDOW_AUTOSIZE)
cv.ShowImage("win", img)
cv.MoveWindow("win", 200, 200)
cv.WaitKey(0)
cv.DestroyAllWindows()
