import numpy as np
import cv2


if __name__ == '__main__':
    import sys
    from itertools import cycle
    from common import draw_str

    try: fn = sys.argv[1]
    except: fn = '../cpp/baboon.jpg'
    img = cv2.imread(fn)

    modes = cycle(['erode/dilate', 'open/close', 'blackhat/tophat', 'gradient'])
    str_modes = cycle(['ellipse', 'rect', 'cross'])
    cur_mode = modes.next()
    cur_str_mode = str_modes.next()

    def update(dummy=None):
        sz = cv2.getTrackbarPos('op/size', 'morphology')
        iters = cv2.getTrackbarPos('iters', 'morphology')
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
        res = cv2.morphologyEx(img, getattr(cv2, oper_name), st, iterations=iters)
        
        draw_str(res, (10, 20), 'mode: ' + cur_mode)
        draw_str(res, (10, 40), 'operation: ' + oper_name)
        draw_str(res, (10, 60), 'structure: ' + str_name)
        draw_str(res, (10, 80), 'ksize: %d  iters: %d' % (sz, iters))
        cv2.imshow('morphology', res)

    cv2.namedWindow('morphology')
    cv2.createTrackbar('op/size', 'morphology', 12, 20, update)
    cv2.createTrackbar('iters', 'morphology', 1, 10, update)
    update()
    print "Controls:"
    print "  1 - change operation"
    print "  2 - change structure element shape"
    print
    while True:
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
        if ch == ord('1'):
            cur_mode = modes.next()
        if ch == ord('2'):
            cur_str_mode = str_modes.next()
        update()
