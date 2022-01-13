#!/usr/bin/env python

import numpy as np
import cv2 as cv

import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source_frame',
        default="")
    parser.add_argument(
        '-dst',
        '--destination_frame',
        default="")


    args = parser.parse_args()

    depth1 = cv.imread(args.source_frame, cv.IMREAD_ANYDEPTH)
    depth2 = cv.imread(args.destination_frame, cv.IMREAD_ANYDEPTH)

    odometry = cv.Odometry()

    Rt = np.zeros((4, 4))

    odometry.compute(depth1, depth2, Rt)

    print(Rt)

if __name__ == '__main__':
    main()
