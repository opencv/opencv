#!/usr/bin/env python

import cv2, re, glob
import numpy as np

def draw_rects(img, rects, color, l = lambda x, y : x + y):
    if rects is not None:
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (l(x1, x2), l(y1, y2)), color, 2)

class Sample:
    def __init__(self, bbs, img):
        self.image = img
        self.bbs = bb

class Detection:
    def __init__(self, bb, conf):
        self.bb = bb
        self.conf = conf

    # we use rect-stype for dt and box style for gt. ToDo: fix it
    def overlap(self, b):
        a = self.bb
        print "HERE:", a, b
        w = min( a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]);
        h = min( a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]);

        cross_area = 0.0 if (w < 0 or h < 0) else float(w * h)
        union_area = (a[2] * a[3]) + ((b[2] - b[0]) * (b[3] - b[1])) - cross_area;

        return cross_area / union_area;


def parse_inria(ipath, f):
    bbs = []
    path = None
    for l in f:
        box = None
        if l.startswith("Bounding box"):
            b = [x.strip() for x in l.split(":")[1].split("-")]
            c = [x[1:-1].split(",") for x in b]
            d = [int(x) for x in sum(c, [])]
            bbs.append(d)

        if l.startswith("Image filename"):
            path = l.split('"')[-2]

    return Sample(path, bbs)

def glob_set(pattern):
    return [__n for __n in glob.iglob(pattern)] #glob.iglob(pattern)

# parse ETH idl file
def parse_idl(f):
    map = {}
    for l in open(f):
        l = re.sub(r"^\"left\/", "{\"", l)
        l = re.sub(r"\:", ":[", l)
        l = re.sub(r"(\;|\.)$", "]}", l)
        map.update(eval(l))
    return map

def match(gts, rects, confs):
    if rects is None:
        return 0

    dts = zip(*[rects.tolist(), confs.tolist()])
    dts = zip(dts[0][0], dts[0][1])
    dts = [Detection(r,c) for r, c in dts]

    for dt in dts:
        for gt in gts:
            overlap =  dt.overlap(gt)
            print overlap