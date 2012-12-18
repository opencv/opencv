'''
Simple test for GPU module
'''

import numpy as np
import cv2

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


img = cv2.imread('../data/aero1.jpg')


iter_n = 100

img2 = cv2.pyrUp(img)  # warm up
t = clock()
for i in xrange(iter_n):
    cv2.pyrUp(img, img2)
print 'CPU time: %.3f s' % (clock()-t)


d_img = cv2.gpu_GpuMat(img)
d_img2 = cv2.gpu_pyrUp(d_img) # warm up
t = clock()
for i in xrange(iter_n):
    cv2.gpu_pyrUp(d_img, d_img2)
d_img2.download() # sync
print 'GPU time: %.3f s' % (clock()-t)

diff = np.abs(img2 - d_img2.download())
print "max |diff| ==", diff.max()  


cv2.imshow('diff*100', diff*100)
cv2.imshow('d_img2', d_img2.download())
cv2.waitKey()
