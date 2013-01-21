#!/usr/bin/env python

import cv2, re, glob
import numpy as np
import matplotlib.pyplot as plt

""" Convert numpy matrices with rectangles and confidences to sorted list of detections."""
def convert2detections(rects, confs, crop_factor = 0.125):
    if rects is None:
        return []

    dts = zip(*[rects.tolist(), confs.tolist()])
    dts = zip(dts[0][0], dts[0][1])
    dts = [Detection(r,c) for r, c in dts]

    dts.sort(lambda x, y : -1  if (x.conf - y.conf) > 0 else 1)
    for dt in dts:
        dt.crop(crop_factor)

    return dts

def cascade(min_scale, max_scale, nscales, f):
    # where we use nms cv::SCascade::DOLLAR == 2
    c = cv2.SCascade(min_scale, max_scale, nscales, 2)
    xml = cv2.FileStorage(f, 0)
    dom = xml.getFirstTopLevelNode()
    assert c.load(dom)
    return c

def cumsum(n):
    cum = []
    y = 0
    for i in n:
        y += i
        cum.append(y)
    return cum

def computeROC(confidenses, tp, nannotated, nframes):
    confidenses, tp = zip(*sorted(zip(confidenses, tp), reverse = True))

    fp = [(1 - x) for x in tp]
    fp = cumsum(fp)
    tp = cumsum(tp)
    miss_rate = [(1 - x / (nannotated + 0.000001)) for x in tp]
    fppi = [x / float(nframes) for x in fp]

    return fppi, miss_rate


def crop_rect(rect, factor):
    val_x = factor * float(rect[2])
    val_y = factor * float(rect[3])
    x = [int(rect[0] + val_x), int(rect[1] + val_y), int(rect[2] - 2.0 * val_x), int(rect[3] - 2.0 * val_y)]
    return x

#

def initPlot():

    fig, ax = plt.subplots()
    fig.canvas.draw()

    plt.xlabel("fppi")
    plt.ylabel("miss rate")
    plt.title("ROC curve Bahnhof")
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')

def showPlot(name):
    plt.savefig(name)
    plt.show()


def plotLogLog(fppi, miss_rate, c):
    plt.semilogy(fppi, miss_rate, color = c, linewidth = 2)


def draw_rects(img, rects, color, l = lambda x, y : x + y):
    if rects is not None:
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (l(x1, x2), l(y1, y2)), color, 2)

def draw_dt(img, dts, color, l = lambda x, y : x + y):
    if dts is not None:
        for dt in dts:
            bb = dt.bb
            x1, y1, x2, y2 = dt.bb[0], dt.bb[1], dt.bb[2], dt.bb[3]

            cv2.rectangle(img, (x1, y1), (l(x1, x2), l(y1, y2)), color, 2)

class Annotation:
    def __init__(self, bb):
        self.bb = bb

class Detection:
    def __init__(self, bb, conf):
        self.bb = bb
        self.conf = conf
        self.matched = False

    def crop(self, factor):
        self.bb = crop_rect(self.bb, factor)

    # we use rect-stype for dt and box style for gt. ToDo: fix it
    def overlap(self, b):

        a = self.bb
        w = min( a[0] + a[2], b[2]) - max(a[0], b[0]);
        h = min( a[1] + a[3], b[3]) - max(a[1], b[1]);

        cross_area = 0.0 if (w < 0 or h < 0) else float(w * h)
        union_area = (a[2] * a[3]) + ((b[2] - b[0]) * (b[3] - b[1])) - cross_area;

        return cross_area / union_area

    def mark_matched(self):
        self.matched = True


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

def norm_box(box, ratio):
    middle = float(box[0] + box[2]) / 2.0
    new_half_width = float(box[3] - box[1]) * ratio / 2.0
    return (int(round(middle - new_half_width)), box[1], int(round(middle + new_half_width)), box[3])


def norm_acpect_ratio(boxes, ratio):
    return [ norm_box(box, ratio)  for box in boxes]


def match(gts, dts):
    # Cartesian product for each detection BB_dt with each BB_gt
    overlaps = [[dt.overlap(gt) for gt in gts]for dt in dts]

    matches_gt = [0]*len(gts)
    matches_dt = [0]*len(dts)

    for idx, row in enumerate(overlaps):
        imax = row.index(max(row))

        if (matches_gt[imax] == 0 and row[imax] > 0.5):
            matches_gt[imax] = 1
            matches_dt[idx]  = 1
    return matches_dt