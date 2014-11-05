#!/usr/bin/env python

import cv2
import numpy as np
import sys


def shift_dft(src, dst=None):
    '''
        Rearrange the quadrants of Fourier image so that the origin is at
        the image center. Swaps quadrant 1 with 3, and 2 with 4.

        src and dst arrays must be equal size & type
    '''

    if dst is None:
        dst = np.empty(src.shape, src.dtype)
    elif src.shape != dst.shape:
        raise ValueError("src and dst must have equal sizes")
    elif src.dtype != dst.dtype:
        raise TypeError("src and dst must have equal types")

    if src is dst:
        ret = np.empty(src.shape, src.dtype)
    else:
        ret = dst

    h, w = src.shape[:2]

    cx1 = cx2 = w/2
    cy1 = cy2 = h/2

    # if the size is odd, then adjust the bottom/right quadrants
    if w % 2 != 0:
        cx2 += 1
    if h % 2 != 0:
        cy2 += 1

    # swap quadrants

    # swap q1 and q3
    ret[h-cy1:, w-cx1:] = src[0:cy1 , 0:cx1 ]   # q1 -> q3
    ret[0:cy2 , 0:cx2 ] = src[h-cy2:, w-cx2:]   # q3 -> q1

    # swap q2 and q4
    ret[0:cy2 , w-cx2:] = src[h-cy2:, 0:cx2 ]   # q2 -> q4
    ret[h-cy1:, 0:cx1 ] = src[0:cy1 , w-cx1:]   # q4 -> q2

    if src is dst:
        dst[:,:] = ret

    return dst

if __name__ == "__main__":

    if len(sys.argv)>1:
        im = cv2.imread(sys.argv[1])
    else :
        im = cv2.imread('../data/baboon.jpg')
        print "usage : python dft.py <image_file>"

    # convert to grayscale
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = im.shape[:2]

    realInput = im.astype(np.float64)

    # perform an optimally sized dft
    dft_M = cv2.getOptimalDFTSize(w)
    dft_N = cv2.getOptimalDFTSize(h)

    # copy A to dft_A and pad dft_A with zeros
    dft_A = np.zeros((dft_N, dft_M, 2), dtype=np.float64)
    dft_A[:h, :w, 0] = realInput

    # no need to pad bottom part of dft_A with zeros because of
    # use of nonzeroRows parameter in cv2.dft()
    cv2.dft(dft_A, dst=dft_A, nonzeroRows=h)

    cv2.imshow("win", im)

    # Split fourier into real and imaginary parts
    image_Re, image_Im = cv2.split(dft_A)

    # Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
    magnitude = cv2.sqrt(image_Re**2.0 + image_Im**2.0)

    # Compute log(1 + Mag)
    log_spectrum = cv2.log(1.0 + magnitude)

    # Rearrange the quadrants of Fourier image so that the origin is at
    # the image center
    shift_dft(log_spectrum, log_spectrum)

    # normalize and display the results as rgb
    cv2.normalize(log_spectrum, log_spectrum, 0.0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow("magnitude", log_spectrum)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
