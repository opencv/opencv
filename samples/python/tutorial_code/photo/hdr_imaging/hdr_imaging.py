from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
import os

def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, 'list.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv.imread(os.path.join(path, tokens[0])))
        times.append(1 / float(tokens[1]))

    return images, np.asarray(times, dtype=np.float32)

parser = argparse.ArgumentParser(description='Code for High Dynamic Range Imaging tutorial.')
parser.add_argument('--input', type=str, help='Path to the directory that contains images and exposure times.')
args = parser.parse_args()

if not args.input:
    parser.print_help()
    exit(0)

## [Load images and exposure times]
images, times = loadExposureSeq(args.input)
## [Load images and exposure times]

## [Estimate camera response]
calibrate = cv.createCalibrateDebevec()
response = calibrate.process(images, times)
## [Estimate camera response]

## [Make HDR image]
merge_debevec = cv.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)
## [Make HDR image]

## [Tonemap HDR image]
tonemap = cv.createTonemap(2.2)
ldr = tonemap.process(hdr)
## [Tonemap HDR image]

## [Perform exposure fusion]
merge_mertens = cv.createMergeMertens()
fusion = merge_mertens.process(images)
## [Perform exposure fusion]

## [Write results]
cv.imwrite('fusion.png', fusion * 255)
cv.imwrite('ldr.png', ldr * 255)
cv.imwrite('hdr.hdr', hdr)
## [Write results]
