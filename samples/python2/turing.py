'''
Multiscale Turing Patterns generator
====================================

Inspired by http://www.jonathanmccabe.com/Cyclic_Symmetric_Multi-Scale_Turing_Patterns.pdf
'''

import numpy as np
import cv2
import cv2.cv as cv
from common import draw_str
import getopt, sys
from itertools import count

help_message = '''
USAGE: turing.py [-o <output.avi>]

Press ESC to stop.
'''

if __name__ == '__main__':
    print help_message

    w, h = 512, 512

    args, args_list = getopt.getopt(sys.argv[1:], 'o:', [])
    args = dict(args)
    out = None
    if '-o' in args:
        fn = args['-o']
        out = cv2.VideoWriter(args['-o'], cv.CV_FOURCC(*'DIB '), 30.0, (w, h), False)
        print 'writing %s ...' % fn

    a = np.zeros((h, w), np.float32)
    cv2.randu(a, np.array([0]), np.array([1]))

    def process_scale(a_lods, lod):
        d = a_lods[lod] - cv2.pyrUp(a_lods[lod+1])
        for i in xrange(lod):
            d = cv2.pyrUp(d)
        v = cv2.GaussianBlur(d*d, (3, 3), 0)
        return np.sign(d), v
        
    scale_num = 6
    for frame_i in count():
        a_lods = [a]
        for i in xrange(scale_num):
            a_lods.append(cv2.pyrDown(a_lods[-1])) 
        ms, vs = [], []
        for i in xrange(1, scale_num):
            m, v = process_scale(a_lods, i)
            ms.append(m)
            vs.append(v)
        mi = np.argmin(vs, 0)
        a += np.choose(mi, ms) * 0.025
        a = (a-a.min()) / a.ptp()

        if out:
            out.write(a)
        vis = a.copy()
        draw_str(vis, (20, 20), 'frame %d' % frame_i)
        cv2.imshow('a', vis)
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows() 			
