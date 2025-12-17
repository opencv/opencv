#!/usr/bin/env python

"""
Prints OpenCV version and build information.

Usage:
    opencv_version.py [--build]

Options:
    --build    Print complete build information
"""

from __future__ import print_function

import argparse
import cv2 as cv


def main():
    parser = argparse.ArgumentParser(
        description="Print OpenCV version and build information"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Print complete build information"
    )

    args = parser.parse_args()

    print("OpenCV version:", cv.__version__)

    if args.build:
        print("\nBuild information:\n")
        print(cv.getBuildInformation())


if __name__ == "__main__":
    main()
