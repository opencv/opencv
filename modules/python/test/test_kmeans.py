#!/usr/bin/env python

'''
K-means clusterization test
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from numpy import random
import sys
PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range

from tests_common import NewOpenCVTests

def make_gaussians(cluster_n, img_size):
    points = []
    ref_distrs = []
    sizes = []
    for i in xrange(cluster_n):
        mean = (0.1 + 0.8*random.rand(2)) * img_size
        a = (random.rand(2, 2)-0.5)*img_size*0.1
        cov = np.dot(a.T, a) + img_size*0.05*np.eye(2)
        n = 100 + random.randint(900)
        pts = random.multivariate_normal(mean, cov, n)
        points.append( pts )
        ref_distrs.append( (mean, cov) )
        sizes.append(n)
    points = np.float32( np.vstack(points) )
    return points, ref_distrs, sizes

def getMainLabelConfidence(labels, nLabels):

    n = len(labels)
    labelsDict = dict.fromkeys(range(nLabels), 0)
    labelsConfDict = dict.fromkeys(range(nLabels))

    for i in range(n):
        labelsDict[labels[i][0]] += 1

    for i in range(nLabels):
        labelsConfDict[i] = float(labelsDict[i]) / n

    return max(labelsConfDict.values())

class kmeans_test(NewOpenCVTests):

    def test_kmeans(self):

        np.random.seed(10)

        cluster_n = 5
        img_size = 512

        points, _, clusterSizes = make_gaussians(cluster_n, img_size)

        term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
        ret, labels, centers = cv2.kmeans(points, cluster_n, term_crit, 10, 0)

        self.assertEqual(len(centers), cluster_n)

        offset = 0
        for i in range(cluster_n):
            confidence = getMainLabelConfidence(labels[offset : (offset + clusterSizes[i])], cluster_n)
            offset += clusterSizes[i]
            self.assertGreater(confidence, 0.9)