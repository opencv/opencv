#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os
import sys
import unittest

from tests_common import NewOpenCVTests

try:

    if sys.version_info[:2] < (3, 0):
        raise unittest.SkipTest('Python 2.x is not supported')

    # FIXME: Need to cover Mosaic & Image as well.
    # FIXME: FText isn't supported yet.
    class gapi_render_test(NewOpenCVTests):
        def __init__(self, *args):
            super().__init__(*args)

            self.size = (300, 300, 3)

            # Rect
            self.rect = (30, 30, 50, 50)
            self.rcolor = (0, 255, 0)
            self.rlt = cv.LINE_4
            self.rthick = 2
            self.rshift = 3

            # Text
            self.text = 'Hello, world!'
            self.org = (100, 100)
            self.ff = cv.FONT_HERSHEY_SIMPLEX
            self.fs = 1.0
            self.tthick = 2
            self.tlt = cv.LINE_8
            self.tcolor = (255, 255, 255)
            self.blo = False

            # Circle
            self.center = (200, 200)
            self.radius = 200
            self.ccolor = (255, 255, 0)
            self.cthick = 2
            self.clt = cv.LINE_4
            self.cshift = 1

            # Line
            self.pt1 = (50, 50)
            self.pt2 = (200, 200)
            self.lcolor = (0, 255, 128)
            self.lthick = 5
            self.llt = cv.LINE_8
            self.lshift = 2

            # Poly
            self.pts = [(50, 100), (100, 200), (25, 250)]
            self.pcolor = (0, 0, 255)
            self.pthick = 3
            self.plt = cv.LINE_4
            self.pshift = 1

        def cvt_nv12_to_yuv(self, y, uv):
            h,w,_ = uv.shape
            upsample_uv = cv.resize(uv, (h * 2, w * 2))
            return cv.merge([y, upsample_uv])

        def cvt_yuv_to_nv12(self, yuv, y_out, uv_out):
            chs = cv.split(yuv, [y_out, None, None])
            uv = cv.merge([chs[1], chs[2]])
            uv_out = cv.resize(uv, (uv.shape[0] // 2, uv.shape[1] // 2), dst=uv_out)
            return y_out, uv_out

        def cvt_bgr_to_yuv_color(self, bgr):
            y = bgr[2] *  0.299000 + bgr[1] *  0.587000 + bgr[0] *  0.114000;
            u = bgr[2] * -0.168736 + bgr[1] * -0.331264 + bgr[0] *  0.500000 + 128;
            v = bgr[2] *  0.500000 + bgr[1] * -0.418688 + bgr[0] * -0.081312 + 128;
            return (y, u, v)

        def test_render_primitives_on_bgr(self):
            expected = np.zeros(self.size, dtype=np.uint8)
            actual = np.array(expected, copy=True)

            # OpenCV
            cv.rectangle(expected, self.rect, self.rcolor, self.rthick, self.rlt, self.rshift)
            cv.putText(expected, self.text, self.org, self.ff, self.fs, self.tcolor, self.tthick, self.tlt, self.blo)
            cv.circle(expected, self.center, self.radius, self.ccolor, self.cthick, self.clt, self.cshift)
            cv.line(expected, self.pt1, self.pt2, self.lcolor, self.lthick, self.llt, self.lshift)
            cv.fillPoly(expected, np.expand_dims(np.array([self.pts]), axis=0), self.pcolor, self.plt, self.pshift)

            # G-API
            g_in = cv.GMat()
            g_prims = cv.GArray.Prim()
            g_out = cv.gapi.wip.draw.render3ch(g_in, g_prims)

            prims = [cv.gapi.wip.draw.Rect(self.rect, self.rcolor, self.rthick, self.rlt, self.rshift),
                     cv.gapi.wip.draw.Text(self.text, self.org, self.ff, self.fs, self.tcolor, self.tthick, self.tlt, self.blo),
                     cv.gapi.wip.draw.Circle(self.center, self.radius, self.ccolor, self.cthick, self.clt, self.cshift),
                     cv.gapi.wip.draw.Line(self.pt1, self.pt2, self.lcolor, self.lthick, self.llt, self.lshift),
                     cv.gapi.wip.draw.Poly(self.pts, self.pcolor, self.pthick, self.plt, self.pshift)]

            comp = cv.GComputation(cv.GIn(g_in, g_prims), cv.GOut(g_out))
            actual = comp.apply(cv.gin(actual, prims))

            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))

        def test_render_primitives_on_nv12(self):
            y_expected = np.zeros((self.size[0], self.size[1], 1), dtype=np.uint8)
            uv_expected = np.zeros((self.size[0] // 2, self.size[1] // 2, 2), dtype=np.uint8)

            y_actual = np.array(y_expected, copy=True)
            uv_actual = np.array(uv_expected, copy=True)

            # OpenCV
            yuv = self.cvt_nv12_to_yuv(y_expected, uv_expected)
            cv.rectangle(yuv, self.rect, self.cvt_bgr_to_yuv_color(self.rcolor), self.rthick, self.rlt, self.rshift)
            cv.putText(yuv, self.text, self.org, self.ff, self.fs, self.cvt_bgr_to_yuv_color(self.tcolor), self.tthick, self.tlt, self.blo)
            cv.circle(yuv, self.center, self.radius, self.cvt_bgr_to_yuv_color(self.ccolor), self.cthick, self.clt, self.cshift)
            cv.line(yuv, self.pt1, self.pt2, self.cvt_bgr_to_yuv_color(self.lcolor), self.lthick, self.llt, self.lshift)
            cv.fillPoly(yuv, np.expand_dims(np.array([self.pts]), axis=0), self.cvt_bgr_to_yuv_color(self.pcolor), self.plt, self.pshift)
            self.cvt_yuv_to_nv12(yuv, y_expected, uv_expected)

            # G-API
            g_y = cv.GMat()
            g_uv = cv.GMat()
            g_prims = cv.GArray.Prim()
            g_out_y, g_out_uv = cv.gapi.wip.draw.renderNV12(g_y, g_uv, g_prims)

            prims = [cv.gapi.wip.draw.Rect(self.rect, self.rcolor, self.rthick, self.rlt, self.rshift),
                     cv.gapi.wip.draw.Text(self.text, self.org, self.ff, self.fs, self.tcolor, self.tthick, self.tlt, self.blo),
                     cv.gapi.wip.draw.Circle(self.center, self.radius, self.ccolor, self.cthick, self.clt, self.cshift),
                     cv.gapi.wip.draw.Line(self.pt1, self.pt2, self.lcolor, self.lthick, self.llt, self.lshift),
                     cv.gapi.wip.draw.Poly(self.pts, self.pcolor, self.pthick, self.plt, self.pshift)]

            comp = cv.GComputation(cv.GIn(g_y, g_uv, g_prims), cv.GOut(g_out_y, g_out_uv))
            y_actual, uv_actual = comp.apply(cv.gin(y_actual, uv_actual, prims))

            self.assertEqual(0.0, cv.norm(y_expected, y_actual, cv.NORM_INF))
            self.assertEqual(0.0, cv.norm(uv_expected, uv_actual, cv.NORM_INF))


except unittest.SkipTest as e:

    message = str(e)

    class TestSkip(unittest.TestCase):
        def setUp(self):
            self.skipTest('Skip tests: ' + message)

        def test_skip():
            pass

    pass

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
