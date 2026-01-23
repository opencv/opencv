## [imports]
import cv2 as cv
import sys
## [imports]

## [imread]
img_path = cv.samples.findFile("starry_night.jpg", required=False)

if not img_path:
    print("Error: sample image 'starry_night.jpg' not found.")
    print(
        "Download it from:\n"
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/starry_night.jpg"
    )
    sys.exit("Sample image not found. See instructions above.")


img = cv.imread(img_path)
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
