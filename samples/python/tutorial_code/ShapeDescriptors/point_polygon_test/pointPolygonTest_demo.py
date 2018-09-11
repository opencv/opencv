from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np

# Create an image
r = 100
src = np.zeros((4*r, 4*r), dtype=np.uint8)

# Create a sequence of points to make a contour
vert = [None]*6
vert[0] = (3*r//2, int(1.34*r))
vert[1] = (1*r, 2*r)
vert[2] = (3*r//2, int(2.866*r))
vert[3] = (5*r//2, int(2.866*r))
vert[4] = (3*r, 2*r)
vert[5] = (5*r//2, int(1.34*r))

# Draw it in src
for i in range(6):
    cv.line(src, vert[i],  vert[(i+1)%6], ( 255 ), 3)

# Get the contours
_, contours, _ = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Calculate the distances to the contour
raw_dist = np.empty(src.shape, dtype=np.float32)
for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        raw_dist[i,j] = cv.pointPolygonTest(contours[0], (j,i), True)

minVal, maxVal, _, maxDistPt = cv.minMaxLoc(raw_dist)
minVal = abs(minVal)
maxVal = abs(maxVal)

# Depicting the  distances graphically
drawing = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        if raw_dist[i,j] < 0:
            drawing[i,j,0] = 255 - abs(raw_dist[i,j]) * 255 / minVal
        elif raw_dist[i,j] > 0:
            drawing[i,j,2] = 255 - raw_dist[i,j] * 255 / maxVal
        else:
            drawing[i,j,0] = 255
            drawing[i,j,1] = 255
            drawing[i,j,2] = 255

cv.circle(drawing,maxDistPt, int(maxVal),tuple(255,255,255), 1, cv.LINE_8, 0)
cv.imshow('Source', src)
cv.imshow('Distance and inscribed circle', drawing)
cv.waitKey()
