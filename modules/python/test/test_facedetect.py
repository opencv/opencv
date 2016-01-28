#!/usr/bin/env python

'''
face detection using haar cascades
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

def intersectionRate(s1, s2):

    x1, y1, x2, y2 = s1
    s1 = [[x1, y1], [x2,y1], [x2, y2], [x1, y2] ]

    x1, y1, x2, y2 = s2
    s2 = [[x1, y1], [x2,y1], [x2, y2], [x1, y2] ]

    area, intersection = cv2.intersectConvexConvex(np.array(s1), np.array(s2))
    return 2 * area / (cv2.contourArea(np.array(s1)) + cv2.contourArea(np.array(s2)))

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

from tests_common import NewOpenCVTests

class facedetect_test(NewOpenCVTests):

    def test_facedetect(self):
        import sys, getopt

        cascade_fn = "../../../data/haarcascades/haarcascade_frontalface_alt.xml"
        nested_fn  = "../../../data/haarcascades/haarcascade_eye.xml"

        cascade = cv2.CascadeClassifier(cascade_fn)
        nested = cv2.CascadeClassifier(nested_fn)

        dirPath = '../../../samples/data/'
        samples = ['lena.jpg', 'kate.jpg']

        faces = []
        eyes = []

        testFaces = [
        #lena
        [[218, 200, 389, 371],
        [ 244, 240, 294, 290],
        [ 309, 246, 352, 289]],

        #kate
        [[207,  89, 436, 318],
        [245, 161, 294, 210],
        [343, 139, 389, 185]]
        ]

        for sample in samples:

            img = cv2.imread(dirPath + sample)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 1.1)

            rects = detect(gray, cascade)
            faces.append(rects)

            if not nested.empty():
                for x1, y1, x2, y2 in rects:
                    roi = gray[y1:y2, x1:x2]
                    subrects = detect(roi.copy(), nested)

                    for rect in subrects:
                        rect[0] += x1
                        rect[2] += x1
                        rect[1] += y1
                        rect[3] += y1

                    eyes.append(subrects)

        faces_matches = 0
        eyes_matches = 0

        eps = 0.8

        for i in range(len(faces)):
            for j in range(len(testFaces)):
                if intersectionRate(faces[i][0], testFaces[j][0]) > eps:
                    faces_matches += 1
                    #check eyes
                    if len(eyes[i]) == 2:
                        if intersectionRate(eyes[i][0], testFaces[j][1]) > eps and intersectionRate(eyes[i][1], testFaces[j][2]):
                            eyes_matches += 1
                        elif intersectionRate(eyes[i][1], testFaces[j][1]) > eps and intersectionRate(eyes[i][0], testFaces[j][2]):
                            eyes_matches += 1

        self.assertEqual(faces_matches, 2)
        self.assertEqual(eyes_matches, 2)