import cv2 as cv
import sys

# Use required=False to prevent a C++ exception crash. 
# It will return an empty string if the file is not found.
img_path = cv.samples.findFile("starry_night.jpg", required=False)

if not img_path:
    print("Could not find 'starry_night.jpg'.")
    print("Please download it from: https://github.com/opencv/opencv/blob/4.x/samples/data/starry_night.jpg")
    sys.exit(1)

img = cv.imread(img_path)

if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("starry_night.png", img)
