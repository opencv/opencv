'''
Feature homography
==================

Example of using features2d framework for interactive video homography matching.

Usage
-----
feature_homography.py [<video source>]

Keys
----
SPACE - set reference frame
ESC   - exit
'''


import numpy as np
import cv2
import video
from common import draw_str, clock
import sys


detector = cv2.FastFeatureDetector(16, True)
detector = cv2.GridAdaptedFeatureDetector(detector)
extractor = cv2.DescriptorExtractor_create('ORB')

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)

green, red = (0, 255, 0), (0, 0, 255)


if __name__ == '__main__':
    print __doc__

    try: src = sys.argv[1]
    except: src = 0
    cap = video.create_capture(src)

    ref_kp = None

    while True:
        ret, img = cap.read()
        vis = img.copy()
        kp = detector.detect(img)
        kp, desc = extractor.compute(img, kp)

        for p in kp:
            x, y = np.int32(p.pt)
            r = int(0.5*p.size)
            cv2.circle(vis, (x, y), r, (0, 255, 0))
        draw_str(vis, (20, 20), 'feature_n: %d' % len(kp))
        
        if ref_kp is not None:
            raw_matches = matcher.knnMatch(desc, 2)
            matches = []
            for m in raw_matches:
                if len(m) == 2:
                    m1, m2 = m
                    if m1.distance < m2.distance * 0.7:
                        matches.append((m1.trainIdx, m1.queryIdx))
            match_n = len(matches)

            inlier_n = 0
            if match_n > 10:
                p0 = np.float32( [ref_kp[i].pt for i, j in matches] )
                p1 = np.float32( [kp[j].pt for i, j in matches] )

                H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 10.0)
                inlier_n = sum(status)
                if inlier_n > 10:
                    for (x1, y1), (x2, y2), inlier in zip(np.int32(p0), np.int32(p1), status):
                        cv2.line(vis, (x1, y1), (x2, y2), (red, green)[inlier])

                    h, w = img.shape[:2]
                    overlay = cv2.warpPerspective(ref_img, H, (w, h))
                    vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0.0)
            draw_str(vis, (20, 40), 'matched: %d ( %d outliers )' % (match_n, match_n-inlier_n))
        
        cv2.imshow('img', vis)
        ch = 0xFF & cv2.waitKey(1)
        if ch == ord(' '):
            matcher.clear()
            matcher.add([desc])
            ref_kp = kp
            ref_img = img.copy()
        if ch == 27:
            break
