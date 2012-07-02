import numpy as np
import cv2
import os
import video
from common import mosaic

from digits import *


def main():
    cap = video.create_capture()

    classifier_fn = 'digits_svm.dat'
    if not os.path.exists(classifier_fn):
        print '"%s" not found, run digits.py first' % classifier_fn
        return 
    
    model = SVM()
    model.load('digits_svm.dat')

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        bin = cv2.medianBlur(bin, 3)
        contours, heirs = cv2.findContours( bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects = map(cv2.boundingRect, contours)
        valid_flags = [ 16 <= h <= 64  and w <= 1.2*h  for x, y, w, h in rects]

        for i, cnt in enumerate(contours):
            if not valid_flags[i]:
                continue
            _, _, _, outer_i = heirs[0, i]
            if outer_i >=0 and valid_flags[outer_i]:
                continue
            x, y, w, h = rects[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
            sub = bin[y:,x:][:h,:w]
            #sub = ~cv2.equalizeHist(sub)
            #_, sub_bin = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            s = 1.5*float(h)/SZ
            m = cv2.moments(sub)
            m00 = m['m00']
            if m00/255 < 0.1*w*h or m00/255 > 0.9*w*h:
                continue

            c1 = np.float32([m['m10'], m['m01']]) / m00
            c0 = np.float32([SZ/2, SZ/2])
            t = c1 - s*c0
            A = np.zeros((2, 3), np.float32)
            A[:,:2] = np.eye(2)*s
            A[:,2] = t
            sub1 = cv2.warpAffine(sub, A, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            sub1 = deskew(sub1)
            if x+w+SZ < frame.shape[1] and y+SZ < frame.shape[0]:
                frame[y:,x+w:][:SZ, :SZ] = sub1[...,np.newaxis]
                
            sample = preprocess_hog([sub1])
            digit = model.predict(sample)[0]
            cv2.putText(frame, '%d'%digit, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)


        cv2.imshow('frame', frame)
        cv2.imshow('bin', bin)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    main()
