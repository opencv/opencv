from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse

## [Load three images with different environment settings]
parser = argparse.ArgumentParser(description='Code for Histogram Comparison tutorial.')
parser.add_argument('--input1', help='Path to input image 1.')
parser.add_argument('--input2', help='Path to input image 2.')
parser.add_argument('--input3', help='Path to input image 3.')
args = parser.parse_args()

src_base = cv.imread(args.input1)
src_test1 = cv.imread(args.input2)
src_test2 = cv.imread(args.input3)
if src_base is None or src_test1 is None or src_test2 is None:
    print('Could not open or find the images!')
    exit(0)
## [Load three images with different environment settings]

## [Convert to HSV]
hsv_base = cv.cvtColor(src_base, cv.COLOR_BGR2HSV)
hsv_test1 = cv.cvtColor(src_test1, cv.COLOR_BGR2HSV)
hsv_test2 = cv.cvtColor(src_test2, cv.COLOR_BGR2HSV)
## [Convert to HSV]

## [Convert to HSV half]
hsv_half_down = hsv_base[hsv_base.shape[0]//2:,:]
## [Convert to HSV half]

## [Using 50 bins for hue and 60 for saturation]
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]

# hue varies from 0 to 179, saturation from 0 to 255
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists

# Use the 0-th and 1-st channels
channels = [0, 1]
## [Using 50 bins for hue and 60 for saturation]

## [Calculate the histograms for the HSV images]
hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_base, hist_base, alpha=1, beta=0, norm_type=cv.NORM_L1)

hist_half_down = cv.calcHist([hsv_half_down], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_half_down, hist_half_down, alpha=1, beta=0, norm_type=cv.NORM_L1)

hist_test1 = cv.calcHist([hsv_test1], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_test1, hist_test1, alpha=1, beta=0, norm_type=cv.NORM_L1)

hist_test2 = cv.calcHist([hsv_test2], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_test2, hist_test2, alpha=1, beta=0, norm_type=cv.NORM_L1)
## [Calculate the histograms for the HSV images]

## [Apply the histogram comparison methods]
for compare_method in range(6):
    base_base = cv.compareHist(hist_base, hist_base, compare_method)
    base_half = cv.compareHist(hist_base, hist_half_down, compare_method)
    base_test1 = cv.compareHist(hist_base, hist_test1, compare_method)
    base_test2 = cv.compareHist(hist_base, hist_test2, compare_method)

    print('Method:', compare_method, 'Perfect, Base-Half, Base-Test(1), Base-Test(2) :',\
          base_base, '/', base_half, '/', base_test1, '/', base_test2)
## [Apply the histogram comparison methods]
