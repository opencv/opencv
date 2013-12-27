import numpy as np
import cv2

img = cv2.imread('data/starry_night.jpg')
img_cl = cv2.ocl_oclMat(img)
img2_cl = cv2.ocl_pyrDown(img_cl)
cv2.imshow('res', img2_cl.download())
cv2.waitKey()
