from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

myHarris_window = 'My Harris corner detector'
myShiTomasi_window = 'My Shi Tomasi corner detector'
myHarris_qualityLevel = 50
myShiTomasi_qualityLevel = 50
max_qualityLevel = 100
rng.seed(12345)

def myHarris_function(val):
    myHarris_copy = np.copy(src)
    myHarris_qualityLevel = max(val, 1)

    for i in range(src_gray.shape[0]):
        for j in range(src_gray.shape[1]):
            if Mc[i,j] > myHarris_minVal + ( myHarris_maxVal - myHarris_minVal )*myHarris_qualityLevel/max_qualityLevel:
                cv.circle(myHarris_copy, (j,i), 4, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)

    cv.imshow(myHarris_window, myHarris_copy)

def myShiTomasi_function(val):
    myShiTomasi_copy = np.copy(src)
    myShiTomasi_qualityLevel = max(val, 1)

    for i in range(src_gray.shape[0]):
        for j in range(src_gray.shape[1]):
            if myShiTomasi_dst[i,j] > myShiTomasi_minVal + ( myShiTomasi_maxVal - myShiTomasi_minVal )*myShiTomasi_qualityLevel/max_qualityLevel:
                cv.circle(myShiTomasi_copy, (j,i), 4, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)

    cv.imshow(myShiTomasi_window, myShiTomasi_copy)

# Load source image and convert it to gray
parser = argparse.ArgumentParser(description='Code for Creating your own corner detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='../data/building.jpg')
args = parser.parse_args()

src = cv.imread(args.input)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Set some parameters
blockSize = 3
apertureSize = 3

# My Harris matrix -- Using cornerEigenValsAndVecs
myHarris_dst = cv.cornerEigenValsAndVecs(src_gray, blockSize, apertureSize)

# calculate Mc
Mc = np.empty(src_gray.shape, dtype=np.float32)
for i in range(src_gray.shape[0]):
    for j in range(src_gray.shape[1]):
        lambda_1 = myHarris_dst[i,j,0]
        lambda_2 = myHarris_dst[i,j,1]
        Mc[i,j] = lambda_1*lambda_2 - 0.04*pow( ( lambda_1 + lambda_2 ), 2 )

myHarris_minVal, myHarris_maxVal, _, _ = cv.minMaxLoc(Mc)

# Create Window and Trackbar
cv.namedWindow(myHarris_window)
cv.createTrackbar('Quality Level:', myHarris_window, myHarris_qualityLevel, max_qualityLevel, myHarris_function)
myHarris_function(myHarris_qualityLevel)

# My Shi-Tomasi -- Using cornerMinEigenVal
myShiTomasi_dst = cv.cornerMinEigenVal(src_gray, blockSize, apertureSize)

myShiTomasi_minVal, myShiTomasi_maxVal, _, _ = cv.minMaxLoc(myShiTomasi_dst)

# Create Window and Trackbar
cv.namedWindow(myShiTomasi_window)
cv.createTrackbar('Quality Level:', myShiTomasi_window, myShiTomasi_qualityLevel, max_qualityLevel, myShiTomasi_function)
myShiTomasi_function(myShiTomasi_qualityLevel)

cv.waitKey()
