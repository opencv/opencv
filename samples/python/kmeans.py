#!/usr/bin/env python

'''
K-means clusterization sample.
Usage:
   kmeans.py

Keyboard shortcuts:
   ESC   - exit
   space - generate new distribution
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from gaussian_mix import make_gaussians

def main():
    cluster_n = 5
    img_size = 512

    # generating bright palette
    colors = np.zeros((1, cluster_n, 3), np.uint8)
    colors[0,:] = 255
    colors[0,:,0] = np.arange(0, 180, 180.0/cluster_n)
    colors = cv.cvtColor(colors, cv.COLOR_HSV2BGR)[0]

    while True:
        print('sampling distributions...')
        points, _ = make_gaussians(cluster_n, img_size)

        term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
        ret, labels, centers = cv.kmeans(points, cluster_n, None, term_crit, 10, 0)

        img = np.zeros((img_size, img_size, 3), np.uint8)
        for (x, y), label in zip(np.int32(points), labels.ravel()):
            c = list(map(int, colors[label]))

            cv.circle(img, (x, y), 1, c, -1)

        cv.imshow('kmeans', img)
        ch = cv.waitKey(0)
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
