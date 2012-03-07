'''
gabor_threads.py
=========

Sample demonstrates:
- use of multiple Gabor filter convolutions to get Fractalius-like image effect (http://www.redfieldplugins.com/filterFractalius.htm)
- use of python threading to accelerate the computation

Usage
-----
gabor_threads.py [image filename]

'''

import numpy as np
import cv2
from threading import Lock
from multiprocessing.pool import ThreadPool
            

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def process_threaded(img, filters, threadn = 8):
    accum = np.zeros_like(img)
    accum_lock = Lock()
    def f(kern):
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        with accum_lock:
            np.maximum(accum, fimg, accum)
    pool = ThreadPool(processes=threadn)
    pool.map(f, filters)
    return accum

if __name__ == '__main__':
    import sys
    from common import Timer
    
    print __doc__
    try: img_fn = sys.argv[1]
    except: img_fn = '../cpp/baboon.jpg'

    img = cv2.imread(img_fn)
    filters = build_filters() 

    with Timer('running single-threaded'):
        res1 = process(img, filters)
    with Timer('running multi-threaded'):
        res2 = process_threaded(img, filters)
    
    print 'res1 == res2: ', (res1 == res2).all()
    cv2.imshow('img', img)
    cv2.imshow('result', res2)
    cv2.waitKey()
