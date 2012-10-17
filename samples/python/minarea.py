#!/usr/bin/python

import urllib2
import cv2.cv as cv
from random import randint

def roundxy(pt):
    return (cv.Round(pt[0]), cv.Round(pt[1]))

def draw_common(points):
    success, center, radius = cv.MinEnclosingCircle(points)
    if success:
        cv.Circle(img, roundxy(center), cv.Round(radius), cv.CV_RGB(255, 255, 0), 1, cv. CV_AA, 0)

    box = cv.MinAreaRect2(points)
    box_vtx = [roundxy(p) for p in cv.BoxPoints(box)]
    cv.PolyLine(img, [box_vtx], 1, cv.CV_RGB(0, 255, 255), 1, cv. CV_AA)

def minarea_array(img, count):
    pointMat = cv.CreateMat(count, 1, cv.CV_32SC2)
    for i in range(count):
        pointMat[i, 0] = (randint(img.width/4, img.width*3/4),
                               randint(img.height/4, img.height*3/4))

    cv.Zero(img)

    for i in range(count):
        cv.Circle(img, roundxy(pointMat[i, 0]), 2, cv.CV_RGB(255, 0, 0), cv.CV_FILLED, cv. CV_AA, 0)

    draw_common(pointMat)

def minarea_seq(img, count, storage):
    points = [(randint(img.width/4, img.width*3/4), randint(img.height/4, img.height*3/4)) for i in range(count)]
    cv.Zero(img)

    for p in points:
        cv.Circle(img, roundxy(p), 2, cv.CV_RGB(255, 0, 0), cv.CV_FILLED, cv. CV_AA, 0)

    draw_common(points)

if __name__ == "__main__":
    img = cv.CreateImage((500, 500), 8, 3)
    storage = cv.CreateMemStorage()

    cv.NamedWindow("rect & circle", 1)

    use_seq = True

    while True:
        count = randint(1, 100)
        if use_seq:
            minarea_seq(img, count, storage)
        else:
            minarea_array(img, count)

        cv.ShowImage("rect & circle", img)
        key = cv.WaitKey() % 0x100
        if key in [27, ord('q'), ord('Q')]:
            break

        use_seq = not use_seq
    cv.DestroyAllWindows()
