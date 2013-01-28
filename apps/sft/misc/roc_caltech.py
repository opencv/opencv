#!/usr/bin/env python

import argparse
import sft

import sys, os, os.path, glob, math, cv2, re
from datetime import datetime
import numpy

if __name__ == "__main__":
    path = "/home/kellan/datasets/caltech/set00/V000.txt"
    # open annotation file
    f = open(path)
    (nFrame, nSample) = sft.caltech.parse_header(f)
    objects = sft.caltech.extract_objects(f)

    caltechSamples = []
    annotations = [[] for i in range(nFrame)]

    for obj in objects:
        (type, start, end) = re.search(r'^lbl=\'(\w+)\'\s+str=(\d+)\s+end=(\d+)\s+hide=0$', obj[0]).groups()
        print type, start, end
        start = int(start) -1
        end   = int(end)
        pos   = sft.caltech.parse_pos(obj[1])
        posv  = sft.caltech.parse_pos(obj[2])
        occl  = sft.caltech.parse_occl(obj[3])

        for idx, (p, pv, oc) in enumerate(zip(*[pos, posv, occl])):
            annotations[start + idx].append((type, p, oc, pv))

    for each in annotations:
        print each
