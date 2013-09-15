#!/usr/bin/env python

import cv2.cv as cv
import numpy as np
cv.NamedWindow('Leak')
while 1:
    leak = np.random.random((480, 640)) * 255
    cv.ShowImage('Leak', leak.astype(np.uint8))
    cv.WaitKey(10)
