from __future__ import print_function
import cv2 as cv
import argparse

## [Load image]
parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
parser.add_argument('--input', help='Path to input image.', default='../data/lena.jpg')
args = parser.parse_args()

src = cv.imread(args.input)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
## [Load image]

## [Convert to grayscale]
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
## [Convert to grayscale]

## [Apply Histogram Equalization]
dst = cv.equalizeHist(src)
## [Apply Histogram Equalization]

## [Display results]
cv.imshow('Source image', src)
cv.imshow('Equalized Image', dst)
## [Display results]

## [Wait until user exits the program]
cv.waitKey()
## [Wait until user exits the program]
