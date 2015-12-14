#!/usr/bin/env python

'''
VideoCapture sample showcasing  some features of the Video4Linux2 backend

Sample shows how VideoCapture class can be used to control parameters
of a webcam such as focus or framerate.
Also the sample provides an example how to access raw images delivered
by the hardware to get a grayscale image in a very efficient fashion.

Keys:
    ESC    - exit
    g      - toggle optimized grayscale conversion

'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, False)  # Known bug: https://github.com/Itseez/opencv/pull/5474

cv2.namedWindow("Video")

convert_rgb = True
fps = int(cap.get(cv2.CAP_PROP_FPS))
focus = int(min(cap.get(cv2.CAP_PROP_FOCUS) * 100, 2**31-1))  # ceil focus to C_LONG as Python3 int can go to +inf

cv2.createTrackbar("FPS", "Video", fps, 30, lambda v: cap.set(cv2.CAP_PROP_FPS, v))
cv2.createTrackbar("Focus", "Video", focus, 100, lambda v: cap.set(cv2.CAP_PROP_FOCUS, v / 100))

while True:
    status, img = cap.read()

    fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not bool(cap.get(cv2.CAP_PROP_CONVERT_RGB)):
        if fourcc == "MJPG":
            img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        elif fourcc == "YUYV":
            img = cv2.cvtColor(img, cv2.COLOR_YUV2GRAY_YUYV)
        else:
            print("unsupported format")
            break

    cv2.putText(img, "Mode: {}".format(fourcc), (15, 40), font, 1.0, color)
    cv2.putText(img, "FPS: {}".format(fps), (15, 80), font, 1.0, color)
    cv2.imshow("Video", img)

    k = 0xFF & cv2.waitKey(1)

    if k == 27:
        break
    elif k == ord("g"):
        convert_rgb = not convert_rgb
        cap.set(cv2.CAP_PROP_CONVERT_RGB, convert_rgb)
