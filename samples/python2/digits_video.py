import numpy as np
import cv2
import digits
import os
import video
from common import mosaic



def main():
    cap = video.create_capture()

    classifier_fn = 'digits_svm.dat'
    if not os.path.exists(classifier_fn):
        print '"%s" not found, run digits.py first' % classifier_fn
        return 
    
    model = digits.SVM()
    model.load('digits_svm.dat')

    SZ = 20

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        bin = cv2.medianBlur(bin, 3)
        contours, _ = cv2.findContours( bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < 16 or h > 60 or 1.2*h < w:
                continue
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
            sub = bin[y:,x:][:h,:w]
            #sub = ~cv2.equalizeHist(sub)
            #_, sub_bin = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            s = float(h)/SZ
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
            sub1 = digits.deskew(sub1)
            if x+w+SZ < frame.shape[1] and y+SZ < frame.shape[0]:
                frame[y:,x+w:][:SZ, :SZ] = sub1[...,np.newaxis]
                
            sample = np.float32(sub1).reshape(1,SZ*SZ) / 255.0
            digit = model.predict(sample)[0]

            cv2.putText(frame, '%d'%digit, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)


        cv2.imshow('frame', frame)
        cv2.imshow('bin', bin)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    main()
