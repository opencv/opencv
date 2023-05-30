## Run "python gray.py" 
import cv2

## Read original RGB image from the local folder
img = cv2.imread("lena.png")

## Show RGB image output
cv2.imshow("Orignal Image", img)
cv2.waitKey(0)

## Convert RGB image to gray scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Show gray image output
cv2.imshow("Gray Image", gray_img)
cv2.waitKey(0)

## Destroy all the output windows
cv2.destroyAllWindows()