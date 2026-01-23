## [imports]
import cv2 as cv
import sys
## [imports]
## [imread]
img = cv.imread(cv.samples.findFile("starry_night.jpg"))
if img is None:
    # If sample image is not available (e.g., pip install), create a sample image
    # This ensures the tutorial works out-of-the-box with pip install opencv-python
    import numpy as np
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    print("Note: Sample image not found. Using generated image for demonstration.")
    print("For the full tutorial with the actual starry_night.jpg, download the OpenCV samples:")
    print("https://github.com/opencv/opencv/tree/master/samples")
## [imread]
## [empty]
if img is None:
    sys.exit("Could not read the image.")
## [empty]
## [imshow]
cv.imshow("Display window", img)
k = cv.waitKey(0)
## [imshow]
## [imsave]
if k == ord("s"):
    cv.imwrite("starry_night.png", img)
## [imsave]
