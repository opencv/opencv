import numpy as np
import cv2, cv
from common import anorm

help_message = '''SURF image match 

USAGE: findobj.py [ <image1> <image2> ]
'''


def match(desc1, desc2, r_threshold = 0.75):
    res = []
    for i in xrange(len(desc1)):
        dist = anorm( desc2 - desc1[i] )
        n1, n2 = dist.argsort()[:2]
        r = dist[n1] / dist[n2]
        if r < r_threshold:
            res.append((i, n1))
    return np.array(res)

def draw_match(img1, img2, p1, p2, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv.CV_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))
    
    if status is None:
        status = np.ones(len(p1), np.bool_)
    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(np.int32(p1), np.int32(p2), status):
        col = [red, green][inlier]
        if not inlier:
            cv2.line(vis, (x1, y1), (x2+w1, y2), col)
        cv2.circle(vis, (x1, y1), 2, col, -1)
        cv2.circle(vis, (x2+w1, y2), 2, col, -1)
    return vis

if __name__ == '__main__':
    import sys
    try: fn1, fn2 = sys.argv[1:3]
    except:
        fn1 = '../c/box.png'
        fn2 = '../c/box_in_scene.png'
    print help_message

    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0)

    surf = cv2.SURF(1000)
    kp1, desc1 = surf.detect(img1, None, False)
    kp2, desc2 = surf.detect(img2, None, False)
    desc1.shape = (-1, surf.descriptorSize())
    desc2.shape = (-1, surf.descriptorSize())
    print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

    m = match(desc1, desc2)
    matched_p1 = np.array([kp1[i].pt for i, j in m])
    matched_p2 = np.array([kp2[j].pt for i, j in m])
    H, status = cv2.findHomography(matched_p1, matched_p2, cv2.RANSAC, 10.0)
    print '%d / %d  inliers/matched' % (np.sum(status), len(status))

    vis = draw_match(img1, img2, matched_p1, matched_p2, status, H)
    cv2.imshow('find_obj SURF', vis)
    cv2.waitKey()