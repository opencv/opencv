'''
Multiscale Turing Patterns generator.
Inspired by http://www.jonathanmccabe.com/Cyclic_Symmetric_Multi-Scale_Turing_Patterns.pdf
'''

import numpy as np
import cv2, cv
from common import draw_str

w, h = 512, 512

a = np.zeros((h, w, 1), np.float32)
cv2.randu(a, np.array([0]), np.array([1]))
a.shape = (h, w)

def process_scale(a_lods, lod):
    d = a_lods[lod] - cv2.pyrUp(a_lods[lod+1])
    for i in xrange(lod):
        d = cv2.pyrUp(d)
    v = cv2.GaussianBlur(d*d, (3, 3), 0)
    return np.sign(d), v
    
print 'Generating AVI file. Press ESC to stop.'
out = cv2.VideoWriter('turing.avi', cv.CV_FOURCC(*'DIB '), 30.0, (w, h), False)

scale_num = 6
frame_num = 1000
for frame_i in xrange(frame_num):
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

    out.write(a)
    vis = a.copy()
    draw_str(vis, (20, 20), 'frame %d / %d' % (frame_i+1, frame_num))
    cv2.imshow('a', vis)
    if cv2.waitKey(5) == 27:
        break
else:
    print 'done'
    cv2.waitKey()