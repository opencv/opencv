'''
Text skewness correction
This tutorial demonstrates how to correct the skewness in a text.
The program takes as input a skewed source image and shows non skewed text.

Usage:
        python text_skewness_correction.py --image "Image path"
'''

import numpy as np
import cv2 as cv
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image file")
    args = vars(parser.parse_args())

    # load the image from disk
    image = cv.imread(cv.samples.findFile(args["image"]))
    if image is None:
        print("can't read image " + args["image"])
        sys.exit(-1)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

    # Applying erode filter to remove random noise
    erosion_size = 1
    element = cv.getStructuringElement(cv.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size) )
    thresh = cv.erode(thresh, element)

    coords = cv.findNonZero(thresh)
    angle = cv.minAreaRect(coords)[-1]
    # the `cv.minAreaRect` function returns values in the
    # range [-90, 0) if the angle is less than -45 we need to add 90 to it
    if angle < -45:
        angle = (90 + angle)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    cv.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    print("[INFO] angle: {:.2f}".format(angle))
    cv.imshow("Input", image)
    cv.imshow("Rotated", rotated)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
