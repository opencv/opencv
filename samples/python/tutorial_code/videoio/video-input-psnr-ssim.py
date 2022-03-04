#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import argparse
import sys

# [get-psnr]
def getPSNR(I1, I2):
    s1 = cv.absdiff(I1, I2) #|I1 - I2|
    s1 = np.float32(s1)     # cannot make a square on 8 bits
    s1 = s1 * s1            # |I1 - I2|^2
    sse = s1.sum()          # sum elements per channel
    if sse <= 1e-10:        # sum channels
        return 0            # for small values return zero
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr
# [get-psnr]

# [get-mssim]
def getMSSISM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    # INITS

    I1 = np.float32(i1) # cannot calculate on one byte large values
    I2 = np.float32(i2)

    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2 # I1 * I2
    # END INITS

    # PRELIMINARY COMPUTING
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)

    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2

    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2

    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2

    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    ssim_map = cv.divide(t3, t1)    # ssim_map =  t3./t1;

    mssim = cv.mean(ssim_map)       # mssim = average of ssim map
    return mssim
# [get-mssim]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--delay", type=int, default=30, help=" Time delay")
    parser.add_argument("-v", "--psnrtriggervalue", type=int, default=30, help="PSNR Trigger Value")
    parser.add_argument("-r", "--ref", type=str, default="Megamind.avi", help="Path to reference video")
    parser.add_argument("-t", "--undertest", type=str, default="Megamind_bugy.avi",
                        help="Path to the video to be tested")
    args = parser.parse_args()

    sourceReference = args.ref
    sourceCompareWith = args.undertest
    delay = args.delay
    psnrTriggerValue = args.psnrtriggervalue

    framenum = -1 # Frame counter

    captRefrnc = cv.VideoCapture(cv.samples.findFileOrKeep(sourceReference))
    captUndTst = cv.VideoCapture(cv.samples.findFileOrKeep(sourceCompareWith))

    if not captRefrnc.isOpened():
        print("Could not open the reference " + sourceReference)
        sys.exit(-1)
    if not captUndTst.isOpened():
        print("Could not open case test " + sourceCompareWith)
        sys.exit(-1)

    refS = (int(captRefrnc.get(cv.CAP_PROP_FRAME_WIDTH)), int(captRefrnc.get(cv.CAP_PROP_FRAME_HEIGHT)))
    uTSi = (int(captUndTst.get(cv.CAP_PROP_FRAME_WIDTH)), int(captUndTst.get(cv.CAP_PROP_FRAME_HEIGHT)))

    if refS != uTSi:
        print("Inputs have different size!!! Closing.")
        sys.exit(-1)

    WIN_UT = "Under Test"
    WIN_RF = "Reference"

    cv.namedWindow(WIN_RF, cv.WINDOW_AUTOSIZE)
    cv.namedWindow(WIN_UT, cv.WINDOW_AUTOSIZE)
    cv.moveWindow(WIN_RF, 400, 0) #750,  2 (bernat =0)
    cv.moveWindow(WIN_UT, refS[0], 0) #1500, 2

    print("Reference frame resolution: Width={} Height={} of nr#: {}".format(refS[0], refS[1],
                                                                             captRefrnc.get(cv.CAP_PROP_FRAME_COUNT)))
    print("PSNR trigger value {}".format(psnrTriggerValue))

    while True: # Show the image captured in the window and repeat
        _, frameReference = captRefrnc.read()
        _, frameUnderTest = captUndTst.read()

        if frameReference is None or frameUnderTest is None:
            print(" < < <  Game over!  > > > ")
            break

        framenum += 1
        psnrv = getPSNR(frameReference, frameUnderTest)
        print("Frame: {}# {}dB".format(framenum, round(psnrv, 3)), end=" ")

        if (psnrv < psnrTriggerValue and psnrv):
            mssimv = getMSSISM(frameReference, frameUnderTest)
            print("MSSISM: R {}% G {}% B {}%".format(round(mssimv[2] * 100, 2), round(mssimv[1] * 100, 2),
                                                     round(mssimv[0] * 100, 2)), end=" ")

        print()

        cv.imshow(WIN_RF, frameReference)
        cv.imshow(WIN_UT, frameUnderTest)

        k = cv.waitKey(delay)
        if k == 27:
            break

    sys.exit(0)


if __name__ == "__main__":
    main()
