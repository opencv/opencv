from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

low = 20
up = 20

def callback_low(val):
    global low
    low = val

def callback_up(val):
    global up
    up = val

def pickPoint(event, x, y, flags, param):
    if event != cv.EVENT_LBUTTONDOWN:
        return

    # Fill and get the mask
    seed = (x, y)
    newMaskVal = 255
    newVal = (120, 120, 120)
    connectivity = 8
    flags = connectivity + (newMaskVal << 8 ) + cv.FLOODFILL_FIXED_RANGE + cv.FLOODFILL_MASK_ONLY

    mask2 = np.zeros((src.shape[0] + 2, src.shape[1] + 2), dtype=np.uint8)
    print('low:', low, 'up:', up)
    cv.floodFill(src, mask2, seed, newVal, (low, low, low), (up, up, up), flags)
    mask = mask2[1:-1,1:-1]

    cv.imshow('Mask', mask)
    Hist_and_Backproj(mask)

def Hist_and_Backproj(mask):
    h_bins = 30
    s_bins = 32
    histSize = [h_bins, s_bins]
    h_range = [0, 180]
    s_range = [0, 256]
    ranges = h_range + s_range # Concat list
    channels = [0, 1]

    # Get the Histogram and normalize it
    hist = cv.calcHist([hsv], channels, mask, histSize, ranges, accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # Get Backprojection
    backproj = cv.calcBackProject([hsv], channels, hist, ranges, scale=1)

    # Draw the backproj
    cv.imshow('BackProj', backproj)

# Read the image
parser = argparse.ArgumentParser(description='Code for Back Projection tutorial.')
parser.add_argument('--input', help='Path to input image.')
args = parser.parse_args()

src = cv.imread(args.input)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Transform it to HSV
hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

# Show the image
window_image = 'Source image'
cv.namedWindow(window_image)
cv.imshow(window_image, src)

# Set Trackbars for floodfill thresholds
cv.createTrackbar('Low thresh', window_image, low, 255, callback_low)
cv.createTrackbar('High thresh', window_image, up, 255, callback_up)
# Set a Mouse Callback
cv.setMouseCallback(window_image, pickPoint)

cv.waitKey()
