#!/usr/bin/env python

import sys, os, os.path, glob, math, cv2, sft
from datetime import datetime
from optparse import OptionParser
import re
import numpy as np

def extractPositive(f, path, opath, octave, min_possible):
    newobj = re.compile("^lbl=\'(\w+)\'\s+str=(\d+)\s+end=(\d+)\s+hide=0$")
    pos    = re.compile("^pos\s=(\[[((\d+\.+\d*)|\s+|\;)]*\])$")
    occl   = re.compile("^occl\s*=(\[[0-1|\s]*\])$")

    whole_mod_w = int(64  * octave) + 2 * int(20 * octave)
    whole_mod_h = int(128 * octave) + 2 * int(20 * octave)

    goNext = 0
    start  = 0
    end    = 0

    person_id = -1;

    boxes = []
    occls = []

    for l in f:
        m = newobj.match(l)
        if m is not None:
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
                        for idx, box in enumerate(boxes):
                            if occls[idx] == 1:
                                continue

                            x = box[0]
                            y = box[1]
                            w = box[2]
                            h = box[3]

                            id = int(start) - 1 + idx
                            file = os.path.join(path, "I0%04d.jpg" % id)

                            if (start + id) >= end or w < 10 or h < min_possible:
                                continue

                            mat = cv2.imread(file)
                            mat_h, mat_w, _ = mat.shape

                            # let default height of person be 96.
                            scale = h / float(96)
                            rel_scale = scale / octave

                            d_w = whole_mod_w * rel_scale
                            d_h = whole_mod_h * rel_scale

                            tb = (d_h - h) / 2.0
                            lr = (d_w - w) / 2.0

                            x = int(round(x - lr))
                            y = int(round(y - tb))

                            w = int(round(w + lr * 2.0))
                            h = int(round(h + tb * 2.0))

                            inner = [max(5, x), max(5, y), min(mat_w - 5, x + w), min(mat_h - 5, y + h) ]
                            cropped = mat[inner[1]:inner[3], inner[0]:inner[2], :]

                            top     = int(max(0, 0 - y))
                            bottom  = int(max(0, y + h - mat_h))
                            left    = int(max(0, 0 - x))
                            right   = int(max(0, x + w - mat_w))

                            if top < -d_h / 4.0 or bottom > d_h / 4.0 or left < -d_w / 4.0 or right > d_w / 4.0:
                                continue

                            cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_REPLICATE)
                            resized = sft.resize_sample(cropped, whole_mod_w, whole_mod_h)
                            flipped = cv2.flip(resized, 1)

                            cv2.imshow("resized", resized)

                            c = cv2.waitKey(20)
                            if c == 27:
                                exit(0)

                            fname = re.sub(r"^.*\/(set[0-1]\d)\/(V0\d\d)\.(seq)/(I\d+).jpg$", "\\1_\\2_\\4", file)
                            fname = os.path.join(opath, fname + "_%04d." % person_id + "png")
                            fname_fl = os.path.join(opath, fname + "_mirror_%04d." % person_id + "png")
                            try:
                                cv2.imwrite(fname, resized)
                                cv2.imwrite(fname_fl, flipped)
                            except:
                                print "something wrong... go next."
                                pass

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", metavar="DIRECTORY", type="string",
                       help="Path to the Caltech dataset folder.")

    parser.add_option("-d", "--output-dir", dest="output", metavar="DIRECTORY", type="string",
                       help="Path to store data", default=".")

    parser.add_option("-o", "--octave", dest="octave", type="float",
                       help="Octave for a dataset to be scaled", default="0.5")

    parser.add_option("-m", "--min-possible", dest="min_possible", type="int",
                       help="Minimum possible height for positive.", default="64")

    (options, args) = parser.parse_args()

    if not options.input:
        parser.error("Caltech dataset folder is required.")

    opath = os.path.join(options.output, datetime.now().strftime("raw_ge64_cr_mirr_ts" + "-%Y-%m-%d-%H-%M-%S"))
    os.mkdir(opath)

    gl = glob.iglob( os.path.join(options.input, "set[0][0]/V0[0-9][0-9].txt"))
    for each in gl:
        path, ext = os.path.splitext(each)
        path = path + ".seq"
        print path
        extractPositive(open(each), path, opath, options.octave, options.min_possible)
