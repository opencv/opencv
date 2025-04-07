#!/usr/bin/env python

'''
This program illustrates the use of findContours and drawContours.
The original image is put up along with the image of drawn contours.

Usage:
    contours.py
A trackbar is put up which controls the contour level from -3 to 3
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

def make_image():
    img = np.zeros((500, 500), np.uint8)
    black, white = 0, 255
    for i in xrange(6):
        dx = int((i%2)*250 - 30)
        dy = int((i/2.)*150)

        if i == 0:
            for j in xrange(11):
                angle = (j+5)*np.pi/21
                c, s = np.cos(angle), np.sin(angle)
                x1, y1 = np.int32([dx+100+j*10-80*c, dy+100-90*s])
                x2, y2 = np.int32([dx+100+j*10-30*c, dy+100-30*s])
                cv.line(img, (x1, y1), (x2, y2), white)

        cv.ellipse( img, (dx+150, dy+100), (100,70), 0, 0, 360, white, -1 )
        cv.ellipse( img, (dx+115, dy+70), (30,20), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+185, dy+70), (30,20), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+115, dy+70), (15,15), 0, 0, 360, white, -1 )
        cv.ellipse( img, (dx+185, dy+70), (15,15), 0, 0, 360, white, -1 )
        cv.ellipse( img, (dx+115, dy+70), (5,5), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+185, dy+70), (5,5), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+150, dy+100), (10,5), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+150, dy+150), (40,10), 0, 0, 360, black, -1 )
        cv.ellipse( img, (dx+27, dy+100), (20,35), 0, 0, 360, white, -1 )
        cv.ellipse( img, (dx+273, dy+100), (20,35), 0, 0, 360, white, -1 )
    return img

def main():
    img = make_image()
    h, w = img.shape[:2]

    contours0, hierarchy = cv.findContours( img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0]

    def update(levels):
        vis = np.zeros((h, w, 3), np.uint8)
        levels = levels - 3
        cv.drawContours( vis, contours, (-1, 2)[levels <= 0], (128,255,255),
            3, cv.LINE_AA, hierarchy, abs(levels) )
        cv.imshow('contours', vis)
    update(3)
    cv.createTrackbar( "levels+3", "contours", 3, 7, update )
    cv.imshow('image', img)
    cv.waitKey()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
