import cv2.cv as cv
import math
import time

N=50000
print "leak4"
while True:
    seq=list((i*1., i*1.) for i in range(N))
    cv.Moments(seq)
