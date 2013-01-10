#!/usr/bin/env python

import sys, os, os.path, glob, math, cv2
from datetime import datetime
from optparse import OptionParser
import re

start_templates = ["lbl", "pos", "occl"]

class Sample:
    def __init__(self, l):
        self

if __name__ == "__main__":
    f = open("/home/kellan/datasets/caltech/set00/V004.txt")
    person = re.compile("^lbl=\'person\'\s+str=(\d+)\s+end=(\d+)\s+hide=0$")
    newobj = re.compile("^lbl=\'(\w+)\'\s+str=(\d+)\s+end=(\d+)\s+hide=0$")
    pos = re.compile("^pos\s=(\[[((\d+\.+\d*)|\s+|\;)]*\])$")
    nonarray = re.compile("\;\s+(?!\])|\s+(?!\])")
    lastSemicolon = re.compile("\;\s+(?=\])")
    qqq = re.compile("(?=\[)\b(?=\d*)")

    goNext = 0
    start = 0
    end = 0

    modelW = 32
    modelH = 64

    for l in f:
        qq = newobj.match(l)
        if qq is not None:
            if qq.group(1) == "person":
                goNext = 1
            else:
                goNext = 0
            print qq.group(0), qq.group(1)
        m = person.match(l)
        if m is not None:
            start = m.group(1)
            end   = m.group(2)

            print m.group(0), start, end
        else:
            m = pos.match(l)
            if m is not None:
                if not goNext:
                    continue
                strarr = re.sub(r"\s", ", ", re.sub(r"\;\s+(?=\])", "]", re.sub(r"\;\s+(?!\])", "],[", re.sub(r"(\[)(\d)", "\\1[\\2", m.group(1)))))
                list = eval(strarr)
                for idx, box in enumerate(list):
                    if (box[2] >= 32) or (box[3] >= 64):
                        x = box[0]
                        y = box[1]
                        w = box[2]
                        h = box[3]

                        ratio = w / h
                        neww = h / 2.0
                        offset = (w - neww) / 2.0
                        print "HERE is big!! ", box, ratio, offset
                        if (x + offset) > 0:
                            id = int(start) + idx
                            file = "/home/kellan/datasets/caltech/set00/V004.seq/I0%04d.jpg" % id # I00000.jpg
                            print file
                            img = cv2.imread(file)
                            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0,255,0), 2)
                            cv2.imshow("sample", img)
                            cv2.waitKey(0)

