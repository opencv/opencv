#!/usr/bin/env python

'''
Morphology operations.
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

import numpy as np
import cv2

from tests_common import NewOpenCVTests

class morphology_test(NewOpenCVTests):

    def test_morphology(self):

        fn = 'samples/data/baboon.jpg'
        img = self.get_sample(fn)

        modes = ['erode/dilate', 'open/close', 'blackhat/tophat', 'gradient']
        str_modes = ['ellipse', 'rect', 'cross']

        referenceHashes = { modes[0]: '1bd14fc814e41b80ce7816bc04f60b65', modes[1] : '1bd14fc814e41b80ce7816bc04f60b65',
            modes[2] : 'cb18a5d28e77522dfec6a6255bc3847e', modes[3] : '84909517e4866aa079f4b2e2906bf47b'}

        def update(cur_mode):
            cur_str_mode = str_modes[0]
            sz = 10
            iters = 1
            opers = cur_mode.split('/')
            if len(opers) > 1:
                sz = sz - 10
                op = opers[sz > 0]
                sz = abs(sz)
            else:
                op = opers[0]
            sz = sz*2+1

            str_name = 'MORPH_' + cur_str_mode.upper()
            oper_name = 'MORPH_' + op.upper()

            st = cv2.getStructuringElement(getattr(cv2, str_name), (sz, sz))
            return cv2.morphologyEx(img, getattr(cv2, oper_name), st, iterations=iters)

        for mode in modes:
            res = update(mode)
            self.assertEqual(self.hashimg(res), referenceHashes[mode])