#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
from numpy import random
import cv2

def make_gaussians(cluster_n, img_size):
    points = []
    ref_distrs = []
    for i in xrange(cluster_n):
        mean = (0.1 + 0.8*random.rand(2)) * img_size
        a = (random.rand(2, 2)-0.5)*img_size*0.1
        cov = np.dot(a.T, a) + img_size*0.05*np.eye(2)
        n = 100 + random.randint(900)
        pts = random.multivariate_normal(mean, cov, n)
        points.append( pts )
        ref_distrs.append( (mean, cov) )
    points = np.float32( np.vstack(points) )
    return points, ref_distrs

from tests_common import NewOpenCVTests

class gaussian_mix_test(NewOpenCVTests):

    def test_gaussian_mix(self):

        np.random.seed(10)
        cluster_n = 5
        img_size = 512

        points, ref_distrs = make_gaussians(cluster_n, img_size)

        em = cv2.ml.EM_create()
        em.setClustersNumber(cluster_n)
        em.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_GENERIC)
        em.trainEM(points)
        means = em.getMeans()
        covs = em.getCovs()  # Known bug: https://github.com/opencv/opencv/pull/4232
        found_distrs = zip(means, covs)

        matches_count = 0

        meanEps = 0.05
        covEps = 0.1

        for i in range(cluster_n):
            for j in range(cluster_n):
                if (cv2.norm(means[i] - ref_distrs[j][0], cv2.NORM_L2) / cv2.norm(ref_distrs[j][0], cv2.NORM_L2) < meanEps and
                    cv2.norm(covs[i] - ref_distrs[j][1], cv2.NORM_L2) / cv2.norm(ref_distrs[j][1], cv2.NORM_L2) < covEps):
                    matches_count += 1

        self.assertEqual(matches_count, cluster_n)