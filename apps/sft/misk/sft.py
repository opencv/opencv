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

def crop_rect(rect, factor):
    val_x = factor * float(rect[2])
    val_y = factor * float(rect[3])
    x = [int(rect[0] + val_x), int(rect[1] + val_y), int(rect[2] - 2.0 * val_x), int(rect[3] - 2.0 * val_y)]
    return x

#

def plot_curve():

    fig, ax = plt.subplots()
    fig.canvas.draw()

    x = np.linspace(pow(10,-4), pow(10,1), 101)
    y = 1 - x

    plt.semilogy(x,y,color='m',linewidth=2)
    plt.xlabel("fppi")
    plt.ylabel("miss rate")
    plt.title("ROC curve Bahnhof")

    plt.yticks( [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.64, 0.80])
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    ax.set_yticklabels( ylabels )
    plt.grid(True)

    # plt.xticks( [pow(10, -4), pow(10, -3), pow(10, -2), pow(10, -1), pow(10, 0), pow(10, 0)])
    # xlabels = [item.get_text() for item in ax.get_xticklabels()]
    # ax.set_xticklabels( xlabels )

    plt.xscale('log')
    plt.show()

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

        print self.bb, "vs", b
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

    for dt in dts:
        print  dt.bb,

    print

    for gt in gts:
        print gt


    # Cartesian product for each detection BB_dt with each BB_gt
    overlaps = [[dt.overlap(gt) for gt in gts]for dt in dts]
    print overlaps

    matches_gt = [0]*len(gts)
    print matches_gt

    matches_dt = [0]*len(dts)
    print matches_dt

    for idx, row in enumerate(overlaps):
        print idx, row

        imax = row.index(max(row))

        if (matches_gt[imax] == 0 and row[imax] > 0.5):
            matches_gt[imax] = 1
            matches_dt[idx]  = 1

    print matches_gt
    print matches_dt

    fp = sum(1 for x in matches_dt if x == 0)
    fn = sum(1 for x in matches_gt if x == 0)

    return fp, fn