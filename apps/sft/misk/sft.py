#!/usr/bin/env python

import cv2, re, glob
import numpy as np
import matplotlib.pyplot as plt

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

class Annotation:
    def __init__(self, bb):
        self.bb = bb

class Detection:
    def __init__(self, bb, conf):
        self.bb = bb
        self.conf = conf
        self.matched = False

    # def crop(self):
    #     rel_scale = self.bb[1] / 128


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

def match(gts, rects, confs):
    if rects is None:
        return 0

    fp = 0
    fn = 0

    dts = zip(*[rects.tolist(), confs.tolist()])
    dts = zip(dts[0][0], dts[0][1])
    dts = [Detection(r,c) for r, c in dts]

    for gt in gts:

        # exclude small
        if gt[2] - gt[0] < 27:
            continue

        matched = False

        for dt in dts:
            # dt.crop()
            overlap =  dt.overlap(gt)
            print dt.bb,  "vs", gt, overlap
            if overlap > 0.5:
                dt.mark_matched()
                matched = True
                print "matched ", dt.bb, gt

        if not matched:
            fn = fn + 1

    print "fn", fn

    for dt in dts:
        if not dt.matched:
            fp = fp + 1

    print "fp", fp
