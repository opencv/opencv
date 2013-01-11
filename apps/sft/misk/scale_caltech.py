#!/usr/bin/env python

import sys, os, os.path, glob, math, cv2
from datetime import datetime
from optparse import OptionParser
import re
import numpy as np

def showPeople(f, path, opath):
    newobj = re.compile("^lbl=\'(\w+)\'\s+str=(\d+)\s+end=(\d+)\s+hide=0$")
    pos    = re.compile("^pos\s=(\[[((\d+\.+\d*)|\s+|\;)]*\])$")
    occl   = re.compile("^occl\s*=(\[[0-1|\s]*\])$")

    goNext = 0
    start = 0
    end = 0

    person_id = -1;

    boxes = []
    occls = []

    for l in f:
        m = newobj.match(l)
        if m is not None:
            print m.group(1)
            if m.group(1) == "person":
                goNext = 1
                start = int(m.group(2))
                end   = int(m.group(3))
                person_id = person_id + 1
                print m.group(1), person_id, start, end
            else:
                goNext = 0
        else:
            m = pos.match(l)
            if m is not None:
                if not goNext:
                    continue
                strarr = re.sub(r"\s", ", ", re.sub(r"\;\s+(?=\])", "]", re.sub(r"\;\s+(?!\])", "],[", re.sub(r"(\[)(\d)", "\\1[\\2", m.group(1)))))
                boxes = eval(strarr)
            else:
                m = occl.match(l)
                if m is not None:
                    occls = eval(re.sub(r"\s+(?!\])", ",", m.group(1)))

                    if len(boxes) > 0 and len(boxes) == len(occls):
                        print len(boxes)
                        for idx, box in enumerate(boxes):
                            color = (8, 107, 255)
                            if occls[idx] == 1:
                                continue
                                # color = (255, 107, 8)
                            x = box[0]
                            y = box[1]
                            w = box[2]
                            h = box[3]
                            id = int(start) - 1 + idx
                            file = os.path.join(path, "I0%04d.jpg" % id)

                            print file

                            if (start + id) < end and w > 5 and h > 47:
                                img = cv2.imread(file)

                                fname = re.sub(r"^.*\/(set[0-1]\d)\/(V0\d\d)\.(seq)/(I\d+).jpg$", "\\1_\\2_\\4", file)#os.path.basename(file)
                                fname = os.path.join(opath, fname + "_%04d." % person_id + "png")
                                try:
                                    print "->", fname
                                    submat = img[int(y):int(y + h), int(x):int(x + w),:]
                                    cv2.imwrite(fname, submat)
                                except:
                                    print "something wrong... go next."
                                    pass
                                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
                                cv2.imshow("person", img)

                                c = cv2.waitKey(10)
                                if c == 27:
                                    exit(0)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", metavar="DIRECTORY", type="string",
                       help="path to the Caltech dataset folder.")

    parser.add_option("-o", "--output", dest="output", metavar="DIRECTORY", type="string",
                       help="path to store data", default=".")

    (options, args) = parser.parse_args()

    if not options.input:
        parser.error("Caltech dataset folder is required.")

    opath = os.path.join(options.output, datetime.now().strftime("raw_ge48-" + "-%Y-%m-%d-%H-%M-%S"))
    os.mkdir(opath)

    gl = glob.iglob( os.path.join(options.input, "set[0-1][0-9]/V0[0-9][0-9].txt"))
    for each in gl:
        path, ext = os.path.splitext(each)
        path = path + ".seq"
        print path
        showPeople(open(each), path, opath)