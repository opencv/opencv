'''
Feature-based image matching sample.

USAGE
     find_obj.py [ <image1> <image2> ]
'''

import numpy as np
import cv2

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing

flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)


def draw_match(img1, img2, p1, p2, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

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
        if inlier:
            cv2.line(vis, (x1, y1), (x2+w1, y2), col)
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2+w1, y2), 2, col, -1)
        else:
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2+w1-r, y2-r), (x2+w1+r, y2+r), col, thickness)
            cv2.line(vis, (x2+w1-r, y2+r), (x2+w1+r, y2-r), col, thickness)
    return vis


if __name__ == '__main__':
    print __doc__
    
    import sys
    try: fn1, fn2 = sys.argv[1:3]
    except:
        fn1 = '../c/box.png'
        fn2 = '../c/box_in_scene.png'

    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0)

    detector = cv2.SIFT()
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
    flann_matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)

    def match_and_draw(matcher, r_threshold = 0.75):
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)
        p1, p2 = [], []
        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * r_threshold:
                m = m[0]
                p1.append( kp1[m.queryIdx].pt )
                p2.append( kp2[m.trainIdx].pt )
        p1, p2 = np.float32((p1, p2))
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
            print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        else:
            H, status = None, None
            print '%d matches found, not enough for homography estimation' % len(p1)

        vis = draw_match(img1, img2, p1, p2, status, H)
        return vis

    print 'bruteforce match:',
    vis_brute = match_and_draw( bf_matcher )
    print 'flann match:',
    vis_flann = match_and_draw( flann_matcher )
    cv2.imshow('find_obj', vis_brute)
    cv2.imshow('find_obj flann', vis_flann)
    0xFF & cv2.waitKey()
    cv2.destroyAllWindows() 			
