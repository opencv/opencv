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

import numpy as np
import cv2 as cv

def main():

    def decode_fourcc(v):
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

    font = cv.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_AUTOFOCUS, False)  # Known bug: https://github.com/opencv/opencv/pull/5474

    cv.namedWindow("Video")

    convert_rgb = True
    fps = int(cap.get(cv.CAP_PROP_FPS))
    focus = int(min(cap.get(cv.CAP_PROP_FOCUS) * 100, 2**31-1))  # ceil focus to C_LONG as Python3 int can go to +inf

    cv.createTrackbar("FPS", "Video", fps, 30, lambda v: cap.set(cv.CAP_PROP_FPS, v))
    cv.createTrackbar("Focus", "Video", focus, 100, lambda v: cap.set(cv.CAP_PROP_FOCUS, v / 100))

    while True:
        _status, img = cap.read()

        fourcc = decode_fourcc(cap.get(cv.CAP_PROP_FOURCC))

        fps = cap.get(cv.CAP_PROP_FPS)

        if not bool(cap.get(cv.CAP_PROP_CONVERT_RGB)):
            if fourcc == "MJPG":
                img = cv.imdecode(img, cv.IMREAD_GRAYSCALE)
            elif fourcc == "YUYV":
                img = cv.cvtColor(img, cv.COLOR_YUV2GRAY_YUYV)
            else:
                print("unsupported format")
                break

        cv.putText(img, "Mode: {}".format(fourcc), (15, 40), font, 1.0, color)
        cv.putText(img, "FPS: {}".format(fps), (15, 80), font, 1.0, color)
        cv.imshow("Video", img)

        k = cv.waitKey(1)

        if k == 27:
            break
        elif k == ord('g'):
            convert_rgb = not convert_rgb
            cap.set(cv.CAP_PROP_CONVERT_RGB, convert_rgb)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
