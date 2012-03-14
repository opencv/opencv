'''
Coherence-enhancing filtering example
=====================================

inspired by  
  Joachim Weickert "Coherence-Enhancing Shock Filters"
  http://www.mia.uni-saarland.de/Publications/weickert-dagm03.pdf
'''

import numpy as np
import cv2

def coherence_filter(img, sigma = 11, str_sigma = 11, blend = 0.5, iter_n = 4):
    h, w = img.shape[:2]

    for i in xrange(iter_n):
        print i,
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eigen = cv2.cornerEigenValsAndVecs(gray, str_sigma, 3)
        eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
        x, y = eigen[:,:,1,0], eigen[:,:,1,1]
        
        gxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=sigma)
        gxy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=sigma)
        gyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=sigma)
        gvv = x*x*gxx + 2*x*y*gxy + y*y*gyy
        m = gvv < 0

        ero = cv2.erode(img, None)
        dil = cv2.dilate(img, None)
        img1 = ero
        img1[m] = dil[m]
        img = np.uint8(img*(1.0 - blend) + img1*blend)
    print 'done'
    return img

    
if __name__ == '__main__':
    import sys
    try: fn = sys.argv[1]
    except: fn = '../cpp/baboon.jpg'

    src = cv2.imread(fn)

    def nothing(*argv):
        pass

    def update():
        sigma = cv2.getTrackbarPos('sigma', 'control')*2+1
        str_sigma = cv2.getTrackbarPos('str_sigma', 'control')*2+1
        blend = cv2.getTrackbarPos('blend', 'control') / 10.0
        print 'sigma: %d  str_sigma: %d  blend_coef: %f' % (sigma, str_sigma, blend)
        dst = coherence_filter(src, sigma=sigma, str_sigma = str_sigma, blend = blend)
        cv2.imshow('dst', dst)

    cv2.namedWindow('control', 0)
    cv2.createTrackbar('sigma', 'control', 9, 15, nothing)
    cv2.createTrackbar('blend', 'control', 7, 10, nothing)
    cv2.createTrackbar('str_sigma', 'control', 9, 15, nothing)


    print 'Press SPACE to update the image\n'

    cv2.imshow('src', src)
    update()
    while True:
        ch = 0xFF & cv2.waitKey()
        if ch == ord(' '):
            update()
        if ch == 27:
            break
