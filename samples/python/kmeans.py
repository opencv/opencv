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
import cv2

from gaussian_mix import make_gaussians

if __name__ == '__main__':
    cluster_n = 5
    img_size = 512

    print(__doc__)

    # generating bright palette
    colors = np.zeros((1, cluster_n, 3), np.uint8)
    colors[0,:] = 255
    colors[0,:,0] = np.arange(0, 180, 180.0/cluster_n)
    colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0]

    while True:
        print('sampling distributions...')
        points, _ = make_gaussians(cluster_n, img_size)

        term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
        ret, labels, centers = cv2.kmeans(points, cluster_n, None, term_crit, 10, 0)

        img = np.zeros((img_size, img_size, 3), np.uint8)
        for (x, y), label in zip(np.int32(points), labels.ravel()):
            c = list(map(int, colors[label]))

            cv2.circle(img, (x, y), 1, c, -1)

        cv2.imshow('gaussian mixture', img)
        ch = 0xFF & cv2.waitKey(0)
        if ch == 27:
            break
    cv2.destroyAllWindows()
