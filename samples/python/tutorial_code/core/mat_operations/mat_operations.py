from __future__ import division
import cv2 as cv
import numpy as np

# Snippet code for Operations with images tutorial (not intended to be run)

def load():
    # Input/Output
    filename = 'img.jpg'
    ## [Load an image from a file]
    img = cv.imread(filename)
    ## [Load an image from a file]

    ## [Load an image from a file in grayscale]
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    ## [Load an image from a file in grayscale]

    ## [Save image]
    cv.imwrite(filename, img)
    ## [Save image]

def access_pixel():
    # Accessing pixel intensity values
    img = np.empty((4,4,3), np.uint8)
    y = 0
    x = 0
    ## [Pixel access 1]
    intensity = img[y,x]
    ## [Pixel access 1]

    ## [Pixel access 3]
    blue = img[y,x,0]
    green = img[y,x,1]
    red = img[y,x,2]
    ## [Pixel access 3]

    ## [Pixel access 5]
    img[y,x] = 128
    ## [Pixel access 5]

def reference_counting():
    # Memory management and reference counting
    ## [Reference counting 2]
    img = cv.imread('image.jpg')
    img1 = np.copy(img)
    ## [Reference counting 2]

    ## [Reference counting 3]
    img = cv.imread('image.jpg')
    sobelx = cv.Sobel(img, cv.CV_32F, 1, 0);
    ## [Reference counting 3]

def primitive_operations():
    img = np.empty((4,4,3), np.uint8)
    ## [Set image to black]
    img[:] = 0
    ## [Set image to black]

    ## [Select ROI]
    smallImg = img[10:110,10:110]
    ## [Select ROI]

    ## [BGR to Gray]
    img = cv.imread('image.jpg')
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ## [BGR to Gray]

    src = np.ones((4,4), np.uint8)
    ## [Convert to CV_32F]
    dst = src.astype(np.float32)
    ## [Convert to CV_32F]

def visualize_images():
    ## [imshow 1]
    img = cv.imread('image.jpg')
    cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
    cv.imshow('image', img)
    cv.waitKey()
    ## [imshow 1]

    ## [imshow 2]
    img = cv.imread('image.jpg')
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(grey, cv.CV_32F, 1, 0)
    # find minimum and maximum intensities
    minVal = np.amin(sobelx)
    maxVal = np.amax(sobelx)
    draw = cv.convertScaleAbs(sobelx, alpha=255.0/(maxVal - minVal), beta=-minVal * 255.0/(maxVal - minVal))
    cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
    cv.imshow('image', draw)
    cv.waitKey()
    ## [imshow 2]
