## [imports]
import cv2 as cv
import sys
## [imports]

## [imread]
img = cv.imread(cv.samples.findFile("starry_night.jpg"))
## [imread]

## [empty]
if img is None:
    sys.exit("Could not read the image.")
## [empty]

## [imshow]
cv.imshow("Display window", img)

# Use a periodic wait loop to prevent the Python kernel from freezing
# and allow the user to close the window via the 'X' button.
while True:
    k = cv.waitKey(100) & 0xFF # Timed wait for 100ms
    
    # Break if window is closed (works on most backends)
    if cv.getWindowProperty("Display window", cv.WND_PROP_VISIBLE) < 1:
        break
        
    # Break if Escape key is pressed
    if k == 27:
        break
## [imshow]

## [imsave]
if k == ord("s"):
    cv.imwrite("starry_night.png", img)
## [imsave]