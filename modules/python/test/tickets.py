import unittest
import random
import time
import math
import sys
import array
import os

import cv2.cv as cv

def find_sample(s):
    for d in ["../samples/c/", "../doc/pics/"]:
        path = os.path.join(d, s)
        if os.access(path, os.R_OK):
            return path
    return s

class TestTickets(unittest.TestCase):

    def test_2542670(self):
        xys = [(94, 121), (94, 122), (93, 123), (92, 123), (91, 124), (91, 125), (91, 126), (92, 127), (92, 128), (92, 129), (92, 130), (92, 131), (91, 132), (90, 131), (90, 130), (90, 131), (91, 132), (92, 133), (92, 134), (93, 135), (94, 136), (94, 137), (94, 138), (95, 139), (96, 140), (96, 141), (96, 142), (96, 143), (97, 144), (97, 145), (98, 146), (99, 146), (100, 146), (101, 146), (102, 146), (103, 146), (104, 146), (105, 146), (106, 146), (107, 146), (108, 146), (109, 146), (110, 146), (111, 146), (112, 146), (113, 146), (114, 146), (115, 146), (116, 146), (117, 146), (118, 146), (119, 146), (120, 146), (121, 146), (122, 146), (123, 146), (124, 146), (125, 146), (126, 146), (126, 145), (126, 144), (126, 143), (126, 142), (126, 141), (126, 140), (127, 139), (127, 138), (127, 137), (127, 136), (127, 135), (127, 134), (127, 133), (128, 132), (129, 132), (130, 131), (131, 130), (131, 129), (131, 128), (132, 127), (133, 126), (134, 125), (134, 124), (135, 123), (136, 122), (136, 121), (135, 121), (134, 121), (133, 121), (132, 121), (131, 121), (130, 121), (129, 121), (128, 121), (127, 121), (126, 121), (125, 121), (124, 121), (123, 121), (122, 121), (121, 121), (120, 121), (119, 121), (118, 121), (117, 121), (116, 121), (115, 121), (114, 121), (113, 121), (112, 121), (111, 121), (110, 121), (109, 121), (108, 121), (107, 121), (106, 121), (105, 121), (104, 121), (103, 121), (102, 121), (101, 121), (100, 121), (99, 121), (98, 121), (97, 121), (96, 121), (95, 121)]

        #xys = xys[:12] + xys[16:]
        pts = cv.CreateMat(len(xys), 1, cv.CV_32SC2)
        for i,(x,y) in enumerate(xys):
            pts[i,0] = (x, y)
        storage = cv.CreateMemStorage()
        hull = cv.ConvexHull2(pts, storage)
        hullp = cv.ConvexHull2(pts, storage, return_points = 1)
        defects = cv.ConvexityDefects(pts, hull, storage)

        vis = cv.CreateImage((1000,1000), 8, 3)
        x0 = min([x for (x,y) in xys]) - 10
        x1 = max([x for (x,y) in xys]) + 10
        y0 = min([y for (y,y) in xys]) - 10
        y1 = max([y for (y,y) in xys]) + 10
        def xform(pt):
            x,y = pt
            return (1000 * (x - x0) / (x1 - x0),
                    1000 * (y - y0) / (y1 - y0))

        for d in defects[:2]:
            cv.Zero(vis)

            # First draw the defect as a red triangle
            cv.FillConvexPoly(vis, [xform(p) for p in d[:3]], cv.RGB(255,0,0))

            # Draw the convex hull as a thick green line
            for a,b in zip(hullp, hullp[1:]):
                cv.Line(vis, xform(a), xform(b), cv.RGB(0,128,0), 3)

            # Draw the original contour as a white line
            for a,b in zip(xys, xys[1:]):
                cv.Line(vis, xform(a), xform(b), (255,255,255))

            self.snap(vis)

    def test_2686307(self):
        lena = cv.LoadImage(find_sample("lena.jpg"), 1)
        dst = cv.CreateImage((512,512), 8, 3)
        cv.Set(dst, (128,192,255))
        mask = cv.CreateImage((512,512), 8, 1)
        cv.Zero(mask)
        cv.Rectangle(mask, (10,10), (300,100), 255, -1)
        cv.Copy(lena, dst, mask)
        self.snapL([lena, dst, mask])
        m = cv.CreateMat(480, 640, cv.CV_8UC1)
        print "ji", m
        print m.rows, m.cols, m.type, m.step

    def snap(self, img):
        self.snapL([img])

    def snapL(self, L):
        for i,img in enumerate(L):
            cv.NamedWindow("snap-%d" % i, 1)
            cv.ShowImage("snap-%d" % i, img)
        cv.WaitKey()
        cv.DestroyAllWindows()

if __name__ == '__main__':
    random.seed(0)
    if len(sys.argv) == 1:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestTickets)
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        suite = unittest.TestSuite()
        suite.addTest(TestTickets(sys.argv[1]))
        unittest.TextTestRunner(verbosity=2).run(suite)
