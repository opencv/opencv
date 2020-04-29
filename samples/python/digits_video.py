#!/usr/bin/env python
'''
Digit recognition from video.

Run digits.py before, to train and save the SVM.

Usage:
  digits_video.py [{camera_id|video_file}]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# built-in modules
import os
import sys

# local modules
import video
from common import mosaic

from digits import *

def main():
    try:
        src = sys.argv[1]
    except:
        src = 0
    cap = video.create_capture(src)

    classifier_fn = 'digits_svm.dat'
    if not os.path.exists(classifier_fn):
        print('"%s" not found, run digits.py first' % classifier_fn)
        return

    model = cv.ml.SVM_load(classifier_fn)

    while True:
        _ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


        bin = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 10)
        bin = cv.medianBlur(bin, 3)
        contours, heirs = cv.findContours( bin.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        try:
            heirs = heirs[0]
        except:
            heirs = []

        for cnt, heir in zip(contours, heirs):
            _, _, _, outer_i = heir
            if outer_i >= 0:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            if not (16 <= h <= 64  and w <= 1.2*h):
                continue
            pad = max(h-w, 0)
            x, w = x - (pad // 2), w + pad
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))

            bin_roi = bin[y:,x:][:h,:w]

            m = bin_roi != 0
            if not 0.1 < m.mean() < 0.4:
                continue
            '''
            gray_roi = gray[y:,x:][:h,:w]
            v_in, v_out = gray_roi[m], gray_roi[~m]
            if v_out.std() > 10.0:
                continue
            s = "%f, %f" % (abs(v_in.mean() - v_out.mean()), v_out.std())
            cv.putText(frame, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)
            '''

            s = 1.5*float(h)/SZ
            m = cv.moments(bin_roi)
            c1 = np.float32([m['m10'], m['m01']]) / m['m00']
            c0 = np.float32([SZ/2, SZ/2])
            t = c1 - s*c0
            A = np.zeros((2, 3), np.float32)
            A[:,:2] = np.eye(2)*s
            A[:,2] = t
            bin_norm = cv.warpAffine(bin_roi, A, (SZ, SZ), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
            bin_norm = deskew(bin_norm)
            if x+w+SZ < frame.shape[1] and y+SZ < frame.shape[0]:
                frame[y:,x+w:][:SZ, :SZ] = bin_norm[...,np.newaxis]

            sample = preprocess_hog([bin_norm])
            digit = model.predict(sample)[1].ravel()
            cv.putText(frame, '%d'%digit, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)


        cv.imshow('frame', frame)
        cv.imshow('bin', bin)
        ch = cv.waitKey(1)
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
