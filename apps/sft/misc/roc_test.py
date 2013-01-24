#!/usr/bin/env python

import argparse
import sft

import sys, os, os.path, glob, math, cv2
from datetime import datetime
import numpy

plot_colors = ['b', 'r', 'g', 'c', 'm']

#      "key"    : (  b,   g,   r)
bgr = { "red"   : (  0,   0, 255),
        "green" : (  0, 255,   0),
        "blue"  : (255,   0 ,  0)}

def range(s):
    try:
        lb, rb = map(int, s.split(','))
        return lb, rb
    except:
        raise argparse.ArgumentTypeError("Must be lb, rb")

def call_parser(f, a):
    return eval( "sft.parse_" + f + "('" + a + "')")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Plot ROC curve using Caltech mathod of per image detection performance estimation.')

    # positional
    parser.add_argument("cascade",     help = "Path to the tested detector.",  nargs='+')
    parser.add_argument("input",       help = "Image sequence pattern.")
    parser.add_argument("annotations", help = "Path to the annotations.")

    # optional
    parser.add_argument("-m", "--min_scale", dest = "min_scale", type = float, metavar= "fl",   help = "Minimum scale to be tested.",               default = 0.4)
    parser.add_argument("-M", "--max_scale", dest = "max_scale", type = float, metavar= "fl",   help = "Maximum scale to be tested.",               default = 5.0)
    parser.add_argument("-o", "--output",    dest = "output",    type = str,   metavar= "path", help = "Path to store resultiong image.",           default = "./roc.png")
    parser.add_argument("-n", "--nscales",   dest = "nscales",   type = int,   metavar= "n",    help = "Prefered count of scales from min to max.", default = 55)

    parser.add_argument("-r", "--scale-range",          dest = "scale_range", type = range,  default = (128 * 0.4, 128 * 2.4))
    parser.add_argument("-e", "--extended-range-ratio", dest = "ext_ratio",   type = float,  default = 1.25)

    # required
    parser.add_argument("-f", "--anttn-format", dest = "anttn_format", choices = ['inria', 'caltech', "idl"], help = "Annotation file for test sequence.", required = True)

    args = parser.parse_args()

    print args.scale_range

    print args.cascade
    # # parse annotations
    sft.initPlot()
    samples = call_parser(args.anttn_format, args.annotations)
    for idx, each in enumerate(args.cascade):
        print each
        cascade = sft.cascade(args.min_scale, args.max_scale, args.nscales, each)
        pattern = args.input
        camera =  cv2.VideoCapture(pattern)

        # for plotting over dataset
        nannotated  = 0
        nframes     = 0

        confidenses = []
        tp          = []
        ignored     = []

        while True:
            ret, img = camera.read()
            if not ret:
                break;

            name = pattern % (nframes,)
            _, tail = os.path.split(name)

            boxes = samples[tail]
            boxes = sft.norm_acpect_ratio(boxes, 0.5)
            boxes = [b for b in boxes if (b[3] - b[1]) > args.scale_range[0] / args.ext_ratio]
            boxes = [b for b in boxes if (b[3] - b[1]) < args.scale_range[1] * args.ext_ratio]

            nannotated = nannotated + len(boxes)
            nframes = nframes + 1
            rects, confs = cascade.detect(img, rois = None)

            if confs is None:
                continue

            dts = sft.convert2detections(rects, confs)

            confs = confs.tolist()[0]
            confs.sort(lambda x, y : -1  if (x - y) > 0 else 1)
            confidenses = confidenses + confs

            matched, skip_list = sft.match(boxes, dts)
            tp = tp + matched
            ignored = ignored + skip_list

            print nframes, nannotated

        fppi, miss_rate = sft.computeROC(confidenses, tp, nannotated, nframes, ignored)
        sft.plotLogLog(fppi, miss_rate, plot_colors[idx])

    sft.showPlot("roc_curve.png")