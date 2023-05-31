from __future__ import print_function
from __future__ import division
import cv2 as cv
import argparse

alpha_slider_max = 100
title_window = 'Linear Blend'

## [on_trackbar]
def on_trackbar(val):
    alpha = val / alpha_slider_max
    beta = ( 1.0 - alpha )
    dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    cv.imshow(title_window, dst)
## [on_trackbar]

parser = argparse.ArgumentParser(description='Code for Adding a Trackbar to our applications tutorial.')
parser.add_argument('--input1', help='Path to the first input image.', default='LinuxLogo.jpg')
parser.add_argument('--input2', help='Path to the second input image.', default='WindowsLogo.jpg')
args = parser.parse_args()

## [load]
# Read images ( both have to be of the same size and type )
src1 = cv.imread(cv.samples.findFile(args.input1))
src2 = cv.imread(cv.samples.findFile(args.input2))
## [load]
if src1 is None:
    print('Could not open or find the image: ', args.input1)
    exit(0)

if src2 is None:
    print('Could not open or find the image: ', args.input2)
    exit(0)

## [window]
cv.namedWindow(title_window)
## [window]

## [create_trackbar]
trackbar_name = 'Alpha x %d' % alpha_slider_max
cv.createTrackbar(trackbar_name, title_window , 0, alpha_slider_max, on_trackbar)
## [create_trackbar]

# Show some stuff
on_trackbar(0)

# Wait until user press some key
cv.waitKey()
