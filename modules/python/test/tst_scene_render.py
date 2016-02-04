#!/usr/bin/env python


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from numpy import pi, sin, cos

import cv2

defaultSize = 512

class TestSceneRender():

    def __init__(self, bgImg = None, **params):
        self.time = 0.0
        self.timeStep = 1.0 / 30.0

        if bgImg != None:
            self.sceneBg = bgImg.copy()
        else:
            self.sceneBg = np.zeros((defaultSize, defaultSize, 3), np.uint8)

        self.w = self.sceneBg.shape[0]
        self.h = self.sceneBg.shape[1]

        self.initialRect = np.array([ (self.h/2, self.w/2), (self.h/2, self.w/2 + self.w/10),
         (self.h/2 + self.h/10, self.w/2 + self.w/10), (self.h/2 + self.h/10, self.w/2)])
        self.currentRect = self.initialRect

    def setInitialRect(self, rect):
        self.initialRect = rect

    def getCurrentRect(self):
        x0, y0 = self.currentRect[0]
        x1, y1 = self.currentRect[2]
        return np.array([x0, y0, x1, y1])

    def getNextFrame(self):
        self.time += self.timeStep
        img = self.sceneBg.copy()

        self.currentRect = self.initialRect + np.int( 30*cos(self.time) + 50*sin(self.time/3))
        cv2.fillConvexPoly(img, self.currentRect, (0, 0, 255))

        return img

    def resetTime(self):
        self.time = 0.0


if __name__ == '__main__':

    backGr = cv2.imread('../../../samples/data/lena.jpg')

    render = TestSceneRender(backGr)

    while True:

        img = render.getNextFrame()
        cv2.imshow('img', img)

        ch = 0xFF & cv2.waitKey(3)
        if  ch == 27:
            break
    cv2.destroyAllWindows()