'''
Feature homography
==================

Example of using features2d framework for interactive video homography matching.

Keys
----
SPACE - set reference frame
ESC   - exit
'''

import numpy as np
import cv2
import video
from common import draw_str

if __name__ == '__main__':

    print __doc__

    detector = cv2.FeatureDetector_create('ORB')
    extractor = cv2.DescriptorExtractor_create('ORB')
    matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming') # 'BruteForce-Hamming' # FlannBased

    ref_desc = None
    ref_kp = None

    green, red = (0, 255, 0), (0, 0, 255)

    cap = video.create_capture(0)
    while True:
        ret, img = cap.read()
        vis = img.copy()
        kp = detector.detect(img)

        for p in kp:
            x, y = np.int32(p.pt)
            r = int(0.5*p.size)
            cv2.circle(vis, (x, y), r, (0, 255, 0))
        draw_str(vis, (20, 20), 'feature_n: %d' % len(kp))
        
        desc = extractor.compute(img, kp)
        if ref_desc is not None:
            raw_matches = matcher.knnMatch(desc, ref_desc, 2)
            eps = 1e-5
            matches = [(m1.trainIdx, m1.queryIdx) for m1, m2 in raw_matches if (m1.distance+eps) / (m2.distance+eps) < 0.7]
            match_n = len(matches)

            inliner_n = 0
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
        ch = cv2.waitKey(1)
        if ch == ord(' '):
            ref_desc = desc
            ref_kp = kp
            ref_img = img.copy()
        if ch == 27:
            break
