import numpy as np
import cv2
import cv2.cv as cv
from common import make_cmap

help_message = '''USAGE: distrans.py [<image>]

Keys:
  ESC   - exit
  v     - toggle voronoi mode
'''

if __name__ == '__main__':
    import sys
    try: fn = sys.argv[1]
    except: fn = '../cpp/fruits.jpg'
    print help_message

    img = cv2.imread(fn, 0)
    cm = make_cmap('jet')
    need_update = True
    voronoi = False

    def update(dummy=None):
        global need_update
        need_update = False
        thrs = cv2.getTrackbarPos('threshold', 'distrans')
        mark = cv2.Canny(img, thrs, 3*thrs)
        dist, labels = cv2.distanceTransformWithLabels(~mark, cv.CV_DIST_L2, 5)
        if voronoi:
            vis = cm[np.uint8(labels)]
        else:
            vis = cm[np.uint8(dist*2)]
        vis[mark != 0] = 255
        cv2.imshow('distrans', vis)

    def invalidate(dummy=None):
        global need_update
        need_update = True

    cv2.namedWindow('distrans') 
    cv2.createTrackbar('threshold', 'distrans', 60, 255, invalidate)
    update()


    while True:
        ch = cv2.waitKey(50)
        if ch == 27:
            break
        if ch == ord('v'):
            voronoi = not voronoi
            print 'showing', ['distance', 'voronoi'][voronoi]
            update()
        if need_update:
            update()
      
