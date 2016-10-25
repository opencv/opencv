#!/usr/bin/env python

''' This is a sample for histogram plotting for RGB images and grayscale images for better understanding of colour distribution

Benefit : Learn how to draw histogram of images
          Get familier with cv2.calcHist, cv2.equalizeHist,cv2.normalize and some drawing functions

Level : Beginner or Intermediate

Functions : 1) hist_curve : returns histogram of an image drawn as curves
            2) hist_lines : return histogram of an image drawn as bins ( only for grayscale images )

Usage : python hist.py <image_file>

Abid Rahman 3/14/12 debug Gary Bradski
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np

bins = np.arange(256).reshape(256,1)

def hist_curve(im):
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv2.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        #print("so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y


if __name__ == '__main__':

    import sys

    if len(sys.argv)>1:
        fname = sys.argv[1]
    else :
        fname = '../data/lena.jpg'
        print("usage : python hist.py <image_file>")

    im = cv2.imread(fname)

    if im is None:
        print('Failed to load image file:', fname)
        sys.exit(1)

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)


    print(''' Histogram plotting \n
    Keymap :\n
    a - show histogram for color image in curve mode \n
    b - show histogram in bin mode \n
    c - show equalized histogram (always in bin mode) \n
    d - show histogram for color image in curve mode \n
    e - show histogram for a normalized image in curve mode \n
    Esc - exit \n
    ''')

    cv2.imshow('image',im)
    while True:
        k = cv2.waitKey(0)&0xFF
        if k == ord('a'):
            curve = hist_curve(im)
            cv2.imshow('histogram',curve)
            cv2.imshow('image',im)
            print('a')
        elif k == ord('b'):
            print('b')
            lines = hist_lines(im)
            cv2.imshow('histogram',lines)
            cv2.imshow('image',gray)
        elif k == ord('c'):
            print('c')
            equ = cv2.equalizeHist(gray)
            lines = hist_lines(equ)
            cv2.imshow('histogram',lines)
            cv2.imshow('image',equ)
        elif k == ord('d'):
            print('d')
            curve = hist_curve(gray)
            cv2.imshow('histogram',curve)
            cv2.imshow('image',gray)
        elif k == ord('e'):
            print('e')
            norm = cv2.normalize(gray, gray, alpha = 0,beta = 255,norm_type = cv2.NORM_MINMAX)
            lines = hist_lines(norm)
            cv2.imshow('histogram',lines)
            cv2.imshow('image',norm)
        elif k == 27:
            print('ESC')
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
