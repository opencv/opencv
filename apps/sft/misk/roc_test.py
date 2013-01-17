#!/usr/bin/env python

import argparse
import sft

import sys, os, os.path, glob, math, cv2
from datetime import datetime
import numpy

def call_parser(f, a):
    return eval( "sft.parse_" + f + "('" + a + "')")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Plot ROC curve using Caltech mathod of per image detection performance estimation.')

    # positional
    parser.add_argument("cascade",     help = "Path to the tested detector.")
    parser.add_argument("input",       help = "Image sequence pattern.")
    parser.add_argument("annotations", help = "Path to the annotations.")

    # optional
    parser.add_argument("-m", "--min_scale", dest = "min_scale", type = float, metavar= "fl",   help = "Minimum scale to be tested.",               default = 0.4)
    parser.add_argument("-M", "--max_scale", dest = "max_scale", type = float, metavar= "fl",   help = "Maximum scale to be tested.",               default = 5.0)
    parser.add_argument("-o", "--output",    dest = "output",    type = str,   metavar= "path", help = "Path to store resultiong image.",           default = "./roc.png")
    parser.add_argument("-n", "--nscales",   dest = "nscales",   type = int,   metavar= "n",    help = "Prefered count of scales from min to max.", default = 55)

    # required
    parser.add_argument("-f", "--anttn-format", dest = "anttn_format", choices = ['inria', 'caltech', "idl"], help = "Annotation file for test sequence.", required = True)

    args = parser.parse_args()

    samples = call_parser(args.anttn_format, args.annotations)

    # where we use nms cv::SCascade::DOLLAR == 2
    cascade = cv2.SCascade(args.min_scale, args.max_scale, args.nscales, 2)
    xml = cv2.FileStorage(args.cascade, 0)
    dom = xml.getFirstTopLevelNode()
    assert cascade.load(dom)

    frame = 0
    pattern = args.input
    camera =  cv2.VideoCapture(args.input)
    while True:
        ret, img = camera.read()
        if not ret:
            break;

        name = pattern % (frame,)
        qq = pattern.format(frame)
        _, tail = os.path.split(name)

        boxes = samples[tail]
        if boxes is not None:
            sft.draw_rects(img, boxes, (255, 0, 0), lambda x, y : y)

        frame = frame + 1
        rects, confs = cascade.detect(img, rois = None)

        fp = sft.match(boxes, rects, confs)

    #     # draw results
        if rects is not None:
            sft.draw_rects(img, rects[0], (0, 255, 0))

        cv2.imshow("result", img);
        if (cv2.waitKey (5) != -1):
            break;