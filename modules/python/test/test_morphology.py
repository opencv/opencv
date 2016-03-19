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

        fn = 'samples/data/rubberwhale1.png'
        img = self.get_sample(fn)

        modes = ['erode/dilate', 'open/close', 'blackhat/tophat', 'gradient']
        str_modes = ['ellipse', 'rect', 'cross']

        referenceHashes = { modes[0]: '071a526425b79e45b4d0d71ef51b0562', modes[1] : '071a526425b79e45b4d0d71ef51b0562',
            modes[2] : '427e89f581b7df1b60a831b1ed4c8618', modes[3] : '0dd8ad251088a63d0dd022bcdc57361c'}

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