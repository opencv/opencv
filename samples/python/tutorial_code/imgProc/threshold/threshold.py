from __future__ import print_function
import cv2 as cv
import argparse
import sys

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'

## [Threshold_Demo]
def Threshold_Demo(val):
    # 0: Binary
    # 1: Binary Inverted
    # 2: Threshold Truncated
    # 3: Threshold to Zero
    # 4: Threshold to Zero Inverted
    threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv.getTrackbarPos(trackbar_value, window_name)
    _, dst = cv.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
    cv.imshow(window_name, dst)
## [Threshold_Demo]

parser = argparse.ArgumentParser(description='Code for Basic Thresholding Operations tutorial.')
# Modernization: Switched default from 'stuff.jpg' to 'smarties.png' [#25635]
parser.add_argument('--input', help='Path to input image.', default='smarties.png')
args = parser.parse_args()

## [load]
# Robustness: Use required=False to prevent runtime errors if sample is missing
img_path = cv.samples.findFile(args.input, required=False)

if not img_path:
    print('Could not find sample image:', args.input)
    sys.exit(-1)

# Load an image
src = cv.imread(img_path)
if src is None:
    print('Could not open or find the image: ', img_path)
    sys.exit(-1)

# Convert the image to Gray
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
## [load]

## [window]
# Create a window to display results
cv.namedWindow(window_name)
## [window]

## [trackbar]
# Create Trackbar to choose type of Threshold
cv.createTrackbar(trackbar_type, window_name , 3, max_type, Threshold_Demo)
# Create Trackbar to choose Threshold value
cv.createTrackbar(trackbar_value, window_name , 0, max_value, Threshold_Demo)
## [trackbar]

# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
cv.waitKey()
