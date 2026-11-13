#!/usr/bin/env python
from __future__ import print_function

import os
import subprocess
import sys
import tempfile

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class Features2D_Tests(NewOpenCVTests):

    def test_find_obj_without_train_descriptors(self):
        sample = self.find_file('samples/python/find_obj.py', required=False)
        rng = np.random.RandomState(42)
        textured = rng.randint(0, 256, (128, 128)).astype(np.uint8)
        blank = np.zeros_like(textured)
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, 'textured.png')
            second = os.path.join(directory, 'blank.png')
            self.assertTrue(cv.imwrite(first, textured))
            self.assertTrue(cv.imwrite(second, blank))
            result = subprocess.run(
                [sys.executable, sample, first, second],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Not enough features for matching', result.stdout)

    def test_issue_13406(self):
        self.assertEqual(True, hasattr(cv, 'drawKeypoints'))
        self.assertEqual(True, hasattr(cv, 'DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS'))
        self.assertEqual(True, hasattr(cv, 'DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS'))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
