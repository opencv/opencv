#!/usr/bin/env python

import cv2, re, glob
import numpy             as np
import matplotlib.pyplot as plt
from itertools import izip

""" Convert numPy matrices with rectangles and confidences to sorted list of detections."""
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

""" Create new instance of soft cascade."""
def cascade(min_scale, max_scale, nscales, f):
    # where we use nms cv::SoftCascadeDetector::DOLLAR == 2
    c = cv2.softcascade_Detector(min_scale, max_scale, nscales, 2)
    xml = cv2.FileStorage(f, 0)
    dom = xml.getFirstTopLevelNode()
    assert c.load(dom)
    return c

""" Compute prefix sum for en array."""
def cumsum(n):
    cum = []
    y = 0
    for i in n:
        y += i
        cum.append(y)
    return cum

""" Compute x and y arrays for ROC plot."""
def computeROC(confidenses, tp, nannotated, nframes, ignored):
    confidenses, tp, ignored = zip(*sorted(zip(confidenses, tp, ignored), reverse = True))

    fp = [(1 - x) for x in tp]
    fp = [(x - y) for x, y in izip(fp, ignored)]

    fp = cumsum(fp)
    tp = cumsum(tp)
    miss_rate = [(1 - x / (nannotated + 0.000001)) for x in tp]
    fppi = [x / float(nframes) for x in fp]

    return fppi, miss_rate

""" Crop rectangle by factor."""
def crop_rect(rect, factor):
    val_x = factor * float(rect[2])
    val_y = factor * float(rect[3])
    x = [int(rect[0] + val_x), int(rect[1] + val_y), int(rect[2] - 2.0 * val_x), int(rect[3] - 2.0 * val_y)]
    return x

""" Initialize plot axises."""
def initPlot(name):
    plt.xlabel("fppi")
    plt.ylabel("miss rate")
    plt.title(name)
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')

""" Draw plot."""
def plotLogLog(fppi, miss_rate, c):
    plt.loglog(fppi, miss_rate, color = c, linewidth = 2)

""" Show resulted plot."""
def showPlot(file_name, labels):
    plt.axis((pow(10, -3), pow(10, 1), .035, 1))
    plt.yticks( [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.64, 0.8, 1], ['.05', '.10', '.20', '.30', '.40', '.50', '.64', '.80', '1'] )
    plt.legend(labels, loc = "lower left")
    plt.savefig(file_name)
    plt.show()

""" Filter true positives and ignored detections for cascade detector output."""
def match(gts, dts):
    matches_gt     = [0]*len(gts)
    matches_dt     = [0]*len(dts)
    matches_ignore = [0]*len(dts)

    if len(gts) == 0:
        return matches_dt, matches_ignore

    # Cartesian product for each detection BB_dt with each BB_gt
    overlaps = [[dt.overlap(gt) for gt in gts]for dt in dts]

    for idx, row in enumerate(overlaps):
        imax = row.index(max(row))

        # try to match ground truth
        if (matches_gt[imax] == 0 and row[imax] > 0.5):
            matches_gt[imax] = 1
            matches_dt[idx]  = 1

    for idx, dt in enumerate(dts):
        # try to math ignored
        if matches_dt[idx] == 0:
            row = gts
            row = [i for i in row if (i[3] - i[1]) < 53 or (i[3] - i[1]) >  256]
            for each in row:
                if dts[idx].overlapIgnored(each) > 0.5:
                    matches_ignore[idx] = 1
    return matches_dt, matches_ignore


""" Draw detections or ground truth on image."""
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

class Detection:
    def __init__(self, bb, conf):
        self.bb = bb
        self.conf = conf
        self.matched = False

    def crop(self, factor):
        self.bb = crop_rect(self.bb, factor)

    # we use rect-style for dt and box style for gt. ToDo: fix it
    def overlap(self, b):

        a = self.bb
        w = min( a[0] + a[2], b[2]) - max(a[0], b[0]);
        h = min( a[1] + a[3], b[3]) - max(a[1], b[1]);

        cross_area = 0.0 if (w < 0 or h < 0) else float(w * h)
        union_area = (a[2] * a[3]) + ((b[2] - b[0]) * (b[3] - b[1])) - cross_area;

        return cross_area / union_area

        # we use rect-style for dt and box style for gt. ToDo: fix it
    def overlapIgnored(self, b):

        a = self.bb
        w = min( a[0] + a[2], b[2]) - max(a[0], b[0]);
        h = min( a[1] + a[3], b[3]) - max(a[1], b[1]);

        cross_area = 0.0 if (w < 0 or h < 0) else float(w * h)
        self_area = (a[2] * a[3]);

        return cross_area / self_area

    def mark_matched(self):
        self.matched = True

"""Parse INPIA annotation format"""
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
    return [__n for __n in glob.iglob(pattern)]

""" Parse ETH idl file. """
def parse_idl(f):
    map = {}
    for l in open(f):
        l = re.sub(r"^\"left\/", "{\"", l)
        l = re.sub(r"\:", ":[", l)
        l = re.sub(r"(\;|\.)$", "]}", l)
        map.update(eval(l))
    return map

""" Normalize detection box to unified aspect ration."""
def norm_box(box, ratio):
    middle = float(box[0] + box[2]) / 2.0
    new_half_width = float(box[3] - box[1]) * ratio / 2.0
    return (int(round(middle - new_half_width)), box[1], int(round(middle + new_half_width)), box[3])

""" Process array of boxes."""
def norm_acpect_ratio(boxes, ratio):
    return [ norm_box(box, ratio)  for box in boxes]

""" Filter detections out of extended range. """
def filter_for_range(boxes, scale_range, ext_ratio):
    boxes = norm_acpect_ratio(boxes, 0.5)
    boxes = [b for b in boxes if (b[3] - b[1]) > scale_range[0] / ext_ratio]
    boxes = [b for b in boxes if (b[3] - b[1]) < scale_range[1] * ext_ratio]
    return boxes

""" Resize sample for training."""
def resize_sample(image, d_w, d_h):
    h, w, _ = image.shape
    if (d_h < h) or (d_w < w):
        ratio = min(d_h / float(h), d_w / float(w))

        kernel_size = int( 5 / (2 * ratio))
        sigma = 0.5 / ratio
        image_to_resize = cv2.filter2D(image, cv2.CV_8UC3, cv2.getGaussianKernel(kernel_size, sigma))
        interpolation_type = cv2.INTER_AREA
    else:
        image_to_resize = image
        interpolation_type = cv2.INTER_CUBIC

    return cv2.resize(image_to_resize,(d_w, d_h), None, 0, 0, interpolation_type)

newobj = re.compile("^lbl=\'(\w+)\'\s+str=(\d+)\s+end=(\d+)\s+hide=0$")

class caltech:
    @staticmethod
    def extract_objects(f):
        objects = []
        tmp = []
        for l in f:
            if newobj.match(l) is not None:
                objects.append(tmp)
                tmp = []
            tmp.append(l)
        return objects[1:]

    @staticmethod
    def parse_header(f):
        _    = f.readline() # skip first line (version string)
        head = f.readline()
        (nFrame, nSample) = re.search(r'nFrame=(\d+) n=(\d+)', head).groups()
        return (int(nFrame), int(nSample))

    @staticmethod
    def parse_pos(l):
        pos = re.match(r'^posv?\s*=(\[[\d\s\.\;]+\])$', l).group(1)
        pos = re.sub(r"(\[)(\d)", "\\1[\\2", pos)
        pos = re.sub(r"\s", ", ", re.sub(r"\;\s+(?=\])", "]", re.sub(r"\;\s+(?!\])", "],[", pos)))
        return eval(pos)

    @staticmethod
    def parse_occl(l):
        occl = re.match(r'^occl\s*=(\[[\d\s\.\;]+\])$', l).group(1)
        occl = re.sub(r"\s(?!\])", ",", occl)
        return eval(occl)

def parse_caltech(f):
    (nFrame, nSample) = caltech.parse_header(f)
    objects = caltech.extract_objects(f)

    annotations = [[] for i in range(nFrame)]
    for obj in objects:
        (type, start, end) = re.search(r'^lbl=\'(\w+)\'\s+str=(\d+)\s+end=(\d+)\s+hide=0$', obj[0]).groups()
        print type, start, end
        start = int(start) -1
        end   = int(end)
        pos   = caltech.parse_pos(obj[1])
        posv  = caltech.parse_pos(obj[2])
        occl  = caltech.parse_occl(obj[3])

        for idx, (p, pv, oc) in enumerate(zip(*[pos, posv, occl])):
            annotations[start + idx].append((type, p, oc, pv))

    return annotations
