#!/usr/bin/env python
'''
Compare resize function with interpolations: cv.INTER_NEAREST, cv.INTER_NEAREST_PIL, PIL.Image.NEAREST

Usage:
  resize_pil.py [<image>]

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import sys

def main():

    try:
        fn = sys.argv[1]
    except:
        fn = 'lena.jpg'

    img_cv_lena = cv.imread(cv.samples.findFile(fn), cv.IMREAD_GRAYSCALE)

    if img_cv_lena is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    img_pil_lena = Image.fromarray(np.array(img_cv_lena))

    img_cv_lena_resized = cv.resize(img_cv_lena, (231,322), interpolation=cv.INTER_NEAREST)
    img_pil_lena_resized = img_pil_lena.resize((231,322), Image.NEAREST)
    assert (np.array(img_cv_lena_resized) != np.array(img_pil_lena_resized)).any()
    print('cv.INTER_NEAREST is not equal to Image.NEAREST')

    resized_images = np.hstack((img_cv_lena_resized, img_pil_lena_resized))
    cv.imshow('LEft is OpenCV INTER_NEAREST, right is PIL Image.NEAREST', resized_images)
    cv.waitKey()

    img_cv_lena_resized = cv.resize(img_cv_lena, (231,322), interpolation=cv.INTER_NEAREST_PIL)
    img_pil_lena_resized = img_pil_lena.resize((231,322), Image.NEAREST)
    assert (np.array(img_cv_lena_resized) == np.array(img_pil_lena_resized)).all()
    print('cv.INTER_NEAREST_PIL is equal to Image.NEAREST')
    resized_images = np.hstack((img_cv_lena_resized, img_pil_lena_resized))
    cv.imshow('LEft is OpenCV INTER_NEAREST_PIL, right is PIL Image.NEAREST', resized_images)

    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
