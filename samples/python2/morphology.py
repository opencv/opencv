import numpy as np
import cv2

img = cv2.imread('../cpp/baboon.jpg', False)

def callback(k):
    k = 2*(k-10)
    st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(k), abs(k)))
    op = cv2.MORPH_BLACKHAT
    if k > 0:
        op = cv2.MORPH_TOPHAT
    res = cv2.morphologyEx(img, op, st)
    cv2.imshow('img', res)

callback(20)
cv2.createTrackbar('k', 'img', 10, 20, callback)


cv2.waitKey()
