#!/usr/bin/env python
import cv2

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = '../cpp/fruits.jpg'

    img = cv2.imread(fn)
    if img is None:
        print 'Failed to load image file:', fn
        sys.exit(1)

    img2 = cv2.logPolar(img, (img.shape[0]/2, img.shape[1]/2), 40, cv2.WARP_FILL_OUTLIERS)
    img3 = cv2.linearPolar(img, (img.shape[0]/2, img.shape[1]/2), 40, cv2.WARP_FILL_OUTLIERS)

    cv2.imshow('before', img)
    cv2.imshow('logpolar', img2)
    cv2.imshow('linearpolar', img3)

    cv2.waitKey(0)
