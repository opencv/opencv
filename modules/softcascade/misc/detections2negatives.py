#!/usr/bin/env python

import sys, os, os.path, glob, math, cv2, string, random
from datetime import datetime
from optparse import OptionParser
import re
import numpy as np
from xml.dom import minidom

def resize(image, d_w, d_h):
    if (d_h < image.shape[0]) or (d_w < image.shape[1]):
        ratio = min(d_h / float(image.shape[0]), d_w / float(image.shape[1]))

        kernel_size = int( 5 / (2 * ratio))
        sigma = 0.5 / ratio
        image_to_resize = cv2.filter2D(image, cv2.CV_8UC3, cv2.getGaussianKernel(kernel_size, sigma))
        interpolation_type = cv2.INTER_AREA
    else:
        image_to_resize = image
        interpolation_type = cv2.INTER_CUBIC

    return cv2.resize(image_to_resize,(d_w, d_h), None, 0, 0, interpolation_type)

def det2negative(xmldoc, opath):
    samples = xmldoc.getElementsByTagName('sample')
    for sample in samples:
        detections = sample.getElementsByTagName('detections')
        detections = minidom.parseString(detections[0].toxml())
        detections = detections.getElementsByTagName("_")
        if len(detections) is not 0:
            path = sample.getElementsByTagName("path")
            path = path[0].firstChild.nodeValue
            mat = cv2.imread(path)
            mat_h, mat_w, _ = mat.shape

            for detection in detections:
                detection = detection.childNodes
                for each in detection:
                    rect = eval(re.sub( r"\b\s\b", ",", re.sub(r"\n", "[", each.nodeValue )) + "]")
                    print rect

                    ratio = 64.0 / rect[3]

                    print rect, ratio
                    mat = resize(mat, int(round(mat_w * ratio)), int(round(mat_h * ratio)))

                    rect[0] = int(round(ratio * rect[0])) - 10
                    rect[1] = int(round(ratio * rect[1])) - 10
                    rect[2] = rect[0] + 32 + 20
                    rect[3] = rect[1] + 64 + 20
                    try:
                        cropped = mat[rect[1]:(rect[3]), rect[0]:(rect[2]), :]
                        img = os.path.join(opath, ''.join(random.choice(string.lowercase) for i in range(8)) + ".png")
                        cr_h, cr_w, _ = cropped.shape
                        if cr_h is 84 and cr_w is 52:
                            cv2.imwrite(img, cropped)
                    except:
                        pass

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", metavar="DIRECTORY", type="string",
                       help="Path to the xml collection folder.")

    parser.add_option("-d", "--output-dir", dest="output", metavar="DIRECTORY", type="string",
                       help="Path to store data", default=".")

    (options, args) = parser.parse_args()

    if not options.input:
        parser.error("Input folder is required.")

    opath = os.path.join(options.output, datetime.now().strftime("negatives" + "-%Y-%m-%d-%H-%M-%S"))
    os.mkdir(opath)

    gl = glob.iglob( os.path.join(options.input, "set[0][0]_V0[0][5].seq.xml"))
    for f in gl:
        print f
        xmldoc = minidom.parse(f)
        det2negative(xmldoc, opath)
