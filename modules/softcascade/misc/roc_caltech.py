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
    annotations = sft.parse_caltech(f)

    for each in annotations:
        print each
