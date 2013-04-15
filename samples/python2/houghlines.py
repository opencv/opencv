#!/usr/bin/python
'''
This example illustrates how to use Hough Transform to find lines
Usage: ./houghlines.py [<image_name>]
image argument defaults to ../cpp/pic1.png
'''
import cv2
import numpy as np
import sys
import math

try:
    fn = sys.argv[1]
except:
    fn = "../cpp/pic1.png"
print __doc__
src = cv2.imread(fn)
dst = cv2.Canny(src, 50, 200)
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

# HoughLines()
# lines = cv2.HoughLines(dst, 1, math.pi/180.0, 50, np.array([]), 0, 0)
# a,b,c = lines.shape
# for i in range(b):
#   rho = lines[0][i][0]
#   theta = lines[0][i][1]
#   a = math.cos(theta)
#   b = math.sin(theta)
#   x0, y0 = a*rho, b*rho
#   pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
#   pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
#   cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

lines = cv2.HoughLinesP(dst, 1, math.pi/180.0, 50, np.array([]), 50, 10)
a,b,c = lines.shape
for i in range(b):
    cv2.line(cdst, (lines[0][i][0], lines[0][i][1]), (lines[0][i][2], lines[0][i][3]), (0, 0, 255), 3, cv2.LINE_AA)

cv2.imshow("source", src)
cv2.imshow("detected lines", cdst)
cv2.waitKey(0)
