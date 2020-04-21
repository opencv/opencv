from __future__ import print_function
import sys

import cv2 as cv
import numpy as np


def print_help():
    print('''
    This program demonstrated the use of the discrete Fourier transform (DFT).
    The dft of an image is taken and it's power spectrum is displayed.
    Usage:
    discrete_fourier_transform.py [image_name -- default lena.jpg]''')


def main(argv):

    print_help()

    filename = argv[0] if len(argv) > 0 else 'lena.jpg'

    I = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    if I is None:
        print('Error opening image')
        return -1
    ## [expand]
    rows, cols = I.shape
    m = cv.getOptimalDFTSize( rows )
    n = cv.getOptimalDFTSize( cols )
    padded = cv.copyMakeBorder(I, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])
    ## [expand]
    ## [complex_and_real]
    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv.merge(planes)         # Add to the expanded another plane with zeros
    ## [complex_and_real]
    ## [dft]
    cv.dft(complexI, complexI)         # this way the result may fit in the source matrix
    ## [dft]
    # compute the magnitude and switch to logarithmic scale
    # = > log(1 + sqrt(Re(DFT(I)) ^ 2 + Im(DFT(I)) ^ 2))
    ## [magnitude]
    cv.split(complexI, planes)                   # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv.magnitude(planes[0], planes[1], planes[0])# planes[0] = magnitude
    magI = planes[0]
    ## [magnitude]
    ## [log]
    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv.add(matOfOnes, magI, magI) #  switch to logarithmic scale
    cv.log(magI, magI)
    ## [log]
    ## [crop_rearrange]
    magI_rows, magI_cols = magI.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows/2)
    cy = int(magI_cols/2)

    q0 = magI[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx+cx, 0:cy]     # Top-Right
    q2 = magI[0:cx, cy:cy+cy]     # Bottom-Left
    q3 = magI[cx:cx+cx, cy:cy+cy] # Bottom-Right

    tmp = np.copy(q0)               # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp

    tmp = np.copy(q1)               # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp
    ## [crop_rearrange]
    ## [normalize]
    cv.normalize(magI, magI, 0, 1, cv.NORM_MINMAX) # Transform the matrix with float values into a
    ## viewable image form(float between values 0 and 1).
    ## [normalize]
    cv.imshow("Input Image"       , I   )    # Show the result
    cv.imshow("spectrum magnitude", magI)
    cv.waitKey()

if __name__ == "__main__":
    main(sys.argv[1:])
