import cv
import math
import time

while True:
    h = cv.CreateHist([40], cv.CV_HIST_ARRAY, [[0,255]], 1)
