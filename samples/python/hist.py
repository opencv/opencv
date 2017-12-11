#!/usr/bin/env python

''' This is a sample for histogram plotting for RGB images and grayscale images for better understanding of colour distribution

Benefit : Learn how to draw histogram of images
          Get familier with cv.calcHist, cv.equalizeHist,cv.normalize and some drawing functions

Level : Beginner or Intermediate

Functions : 1) hist_curve : returns histogram of an image drawn as curves
            2) hist_lines : return histogram of an image drawn as bins ( only for grayscale images )

Usage : python hist.py <image_file>

Abid Rahman 3/14/12 debug Gary Bradski
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np

bins = np.arange(256).reshape(256,1)

def hist_curve(im):
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv.calcHist([im],[ch],None,[256],[0,256])
        cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        #print("so converting image to grayscale for representation"
        im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    hist_item = cv.calcHist([im],[0],None,[256],[0,256])
    cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y


if __name__ == '__main__':

    import sys

    if len(sys.argv)>1:
        fname = sys.argv[1]
    else :
        fname = '../data/lena.jpg'
        print("usage : python hist.py <image_file>")

    im = cv.imread(fname)

    if im is None:
        print('Failed to load image file:', fname)
        sys.exit(1)

    gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)


    print(''' Histogram plotting \n
    Keymap :\n
    a - show histogram for color image in curve mode \n
    b - show histogram in bin mode \n
    c - show equalized histogram (always in bin mode) \n
    d - show histogram for color image in curve mode \n
    e - show histogram for a normalized image in curve mode \n
    Esc - exit \n
    ''')

    cv.imshow('image',im)
    while True:
        k = cv.waitKey(0)
        if k == ord('a'):
            curve = hist_curve(im)
            cv.imshow('histogram',curve)
            cv.imshow('image',im)
            print('a')
        elif k == ord('b'):
            print('b')
            lines = hist_lines(im)
            cv.imshow('histogram',lines)
            cv.imshow('image',gray)
        elif k == ord('c'):
            print('c')
            equ = cv.equalizeHist(gray)
            lines = hist_lines(equ)
            cv.imshow('histogram',lines)
            cv.imshow('image',equ)
        elif k == ord('d'):
            print('d')
            curve = hist_curve(gray)
            cv.imshow('histogram',curve)
            cv.imshow('image',gray)
        elif k == ord('e'):
            print('e')
            norm = cv.normalize(gray, gray, alpha = 0,beta = 255,norm_type = cv.NORM_MINMAX)
            lines = hist_lines(norm)
            cv.imshow('histogram',lines)
            cv.imshow('image',norm)
        elif k == 27:
            print('ESC')
            cv.destroyAllWindows()
            break
    cv.destroyAllWindows()
