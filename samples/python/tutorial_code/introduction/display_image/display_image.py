import cv2 as cv
import sys

# Try to find the file using OpenCV's internal search path
# Note: This often fails in pip-installed versions if samples are not included
try:
    img_path = cv.samples.findFile("starry_night.jpg")
except cv.error:
    # If not found, provide a clear instruction instead of a crash
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
