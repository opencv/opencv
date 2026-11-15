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
k = cv.waitKey(0)
## [imshow]
## [imsave]
if k == ord("s"):
    cv.imwrite("starry_night.png", img)
## [imsave]
# [gui_loop]
# To prevent the Python kernel from freezing, it is often better to use a non-blocking 
# waitKey and manually check if the window was closed via the 'X' button.
# Here is a more robust way to display images in a loop:
#
# import time
# cv.namedWindow("Display window", cv.WINDOW_KEEPRATIO)
# cv.imshow("Display window", img)
# 
# start_time = time.time()
# while True:
#     # Check every 100ms if a key is pressed or window closed
#     k = cv.waitKey(100) 
#     if k == 27: # Escape key
#         break
#     # Check if window was closed by X button
#     if cv.getWindowProperty("Display window", cv.WND_PROP_VISIBLE) < 1:
#         break
# cv.destroyAllWindows()
# [gui_loop]