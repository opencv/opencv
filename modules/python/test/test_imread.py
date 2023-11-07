#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np


if __name__ == '__main__':
 
img = np.zeros((1000, 1000, 3), dtype=np.uint8)
subImg = img[64:576, 128:640]
subImg = cv2.imread("../cv/shared/lena.png", subImg)
ori = cv2.imread("../cv/shared/lena.png")

assert cv2.norm(ori, subImg, cv2.NORM_INF) == 0
assert cv2.norm(np.sum(img[0:1000, 0:128])) == 0
assert cv2.norm(np.sum(img[0:1000, 640:1000])) == 0
assert cv2.norm(np.sum(img[0:64, 128:640])) == 0
assert cv2.norm(np.sum(img[576:1000, 128:640])) == 0
