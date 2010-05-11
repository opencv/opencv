#! /usr/bin/env python

import opencv
from opencv import highgui

cap = highgui.cvCreateFileCapture("../c/tree.avi")
img = highgui.cvQueryFrame(cap)
print "Got frame of dimensions (", img.width, " x ", img.height, " )"

highgui.cvNamedWindow("win", highgui.CV_WINDOW_AUTOSIZE)
highgui.cvShowImage("win", img)
highgui.cvMoveWindow("win", 200, 200)
highgui.cvWaitKey(0)

