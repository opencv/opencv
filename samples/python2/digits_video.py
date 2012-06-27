import numpy as np
import cv2
#import video
import digits
from common import mosaic

#cap = video.create_capture()
cap = cv2.VideoCapture(0)

model = digits.SVM()
model.load('digits_svm.dat')

SZ = 20

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    bin = cv2.medianBlur(bin, 3)
    contours, _ = cv2.findContours( bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 20 or h > 60 or 1.2*h < w:
            continue
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
        sub = bin[y:,x:][:h,:w]
        #sub = ~cv2.equalizeHist(sub)
        #_, sub_bin = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        s = 1.1*h/SZ
        m = cv2.moments(sub)
        m00 = m['m00']
        if m00/255 < 0.1*w*h or m00/255 > 0.9*w*h:
            continue

        #frame[y:,x:][:h,:w] = sub[...,np.newaxis]
        c1 = np.float32([m['m10'], m['m01']]) / m00
        c0 = np.float32([SZ/2, SZ/2])
        t = c1 - s*c0
        A = np.zeros((2, 3), np.float32)
        A[:,:2] = np.eye(2)*2
        A[:,2] = t
        sub1 = cv2.warpAffine(sub, A, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        sub1 = digits.deskew(sub1)
        sample = np.float32(sub1).reshape(1,SZ*SZ) / 255.0
        digit = model.predict(sample)[0]

        cv2.putText(frame, '%d'%digit, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)

        boxes.append(sub1)


    if len(boxes) > 0:
        cv2.imshow('box', mosaic(10, boxes))
        

    cv2.imshow('frame', frame)
    cv2.imshow('bin', bin)
    if cv2.waitKey(1) == 27:
        break
