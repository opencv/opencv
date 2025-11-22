#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html

import argparse
import sys
import cv2 as cv

USAGE = """\
Chromatic Aberration Correction Sample
Usage:
  chromatic_aberration_correction.py <input_image> <calibration_file> [--bayer <code>] [--output <path>]

Arguments:
  input_image       Path to the input image. Can be:
                      • a 3-channel BGR image, or
                      • a 1-channel raw Bayer image (see bayer_pattern)
  calibration_file  OpenCV YAML/XML file with chromatic aberration calibration:
                      image_width, image_height, red_channel/coeffs_x, coeffs_y,
                      blue_channel/coeffs_x, coeffs_y.
  output            (optional) Path to save the corrected image. Default: corrected.png
  bayer             (optional) integer code for demosaicing a 1-channel raw image
                    If omitted or <0, input is assumed 3-channel BGR.

Example:
  python chromatic_aberration_correction.py input.png calib.yaml --bayer 46 --output corrected.png
"""

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Chromatic Aberration Correction Sample",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE
    )
    parser.add_argument("input", help="Input image (BGR or Bayer)")
    parser.add_argument("calibration", help="Calibration file (YAML/XML)")
    parser.add_argument("--output", default="corrected.png", help="Output image file")
    parser.add_argument("--bayer", type=int, default=-1, help="Bayer pattern code for demosaic")
    parser.add_argument("--no-gui", action="store_true", help="Do not open image windows")

    args = parser.parse_args(argv)

    img = cv.imread(args.input, cv.IMREAD_UNCHANGED)
    if img is None:
        print(f"ERROR: Could not load input image: {args.input}", file=sys.stderr)
        return 1

    try:
        coeffMat, size, degree = cv.loadCalibrationResultFromFile(args.calibration)
        corrected = cv.correctChromaticAberration(img, coeffMat, size, degree, args.bayer)

        if corrected is None:
            print("ERROR: cv.correctChromaticAberration returned None", file=sys.stderr)
            return 1

        if not args.no_gui:
            cv.namedWindow("Original", cv.WINDOW_AUTOSIZE)
            cv.namedWindow("Corrected", cv.WINDOW_AUTOSIZE)
            cv.imshow("Original", img)
            cv.imshow("Corrected", corrected)
            print("Press any key to continue...")
            cv.waitKey(0)
            cv.destroyAllWindows()

        if not cv.imwrite(args.output, corrected):
            print(f"WARNING: Could not write output image: {args.output}", file=sys.stderr)
        else:
            print(f"Saved corrected image to: {args.output}")

    except cv.error as e:
        print(f"OpenCV error: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
