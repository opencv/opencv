#!/usr/bin/env python

import cv2, re, glob

def draw_rects(img, rects, color, l = lambda x, y : x + y):
    if rects is not None:
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (l(x1, x2), l(y1, y2)), color, 2)

class Sample:
    def __init__(self, bbs, img):
        self.image = img
        self.bbs = bb

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