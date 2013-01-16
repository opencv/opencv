#!/usr/bin/env python

import sys, os, os.path, glob, math, cv2
from datetime import datetime
from optparse import OptionParser
import numpy

def draw_rects(img, rects, color):
    if rects is None:
        return
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2 + x1, y2 + y1), color, 2)

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-i", "--input", dest="input", type="string",
                       help="Image sequence pattern.")

    parser.add_option("-c", "--cascade", dest="cascade",  metavar="FILE", type="string",
                       help="Path to the tested detector.")

    parser.add_option("-m", "--min_scale", dest="min_scale", type = "float",
                       help="Minimum scale to be tested.", default = 0.4)

    parser.add_option("-M", "--max_scale", dest="max_scale", type = "float",
                       help="Maximum scale to be tested.", default = 5.0)

    parser.add_option("-o", "--output", dest="output", metavar="FILE", type="string",
                       help="Path to store resultion image.", default="./roc.png")

    parser.add_option("-n", "--nscales", dest="nscales", type="int",
                       help="Prefered count of scales that should be tested from min to max.", default = 55)

    (options, args) = parser.parse_args()

    if not options.input:
        parser.error("Test sequence is requared.")

    if not options.cascade:
        parser.error("Xml cascade file is requared.")

    # where we use nms cv::SCascade::DOLLAR == 2
    cascade = cv2.SCascade(options.min_scale, options.max_scale, options.nscales, 2)
    xml = cv2.FileStorage(options.cascade, 0)
    xml1 = xml.getFirstTopLevelNode()

    cascade.load(xml1)

    camera =  cv2.VideoCapture(options.input);
    while True:
        ret, img = camera.read();
        if not ret:
            break;

        rects, confs = cascade.detect(img, rois = None)

        # draw results
        if rects is not None:
            draw_rects(img, rects[0], (0, 255, 0))
        cv2.imshow("result",img);
        if (cv2.waitKey (5) != -1):
            break;