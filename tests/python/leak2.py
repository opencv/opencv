import cv
import numpy as np
import time

while True:
    for i in range(4000):
        a = cv.CreateImage((1024,1024), cv.IPL_DEPTH_8U, 1)
        b = cv.CreateMat(1024, 1024, cv.CV_8UC1)
        c = cv.CreateMatND([1024,1024], cv.CV_8UC1)
    print "pause..."
