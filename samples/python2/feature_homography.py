'''
Feature homography
==================

Example of using features2d framework for interactive video homography matching.
ORB features and FLANN matcher are used.

Inspired by http://www.youtube.com/watch?v=-ZNYoL8rzPY

Usage
-----
feature_homography.py [<video source>]

Select a textured planar object to track by drawing a box with a mouse.

'''

import numpy as np
import cv2
import video
import common
from operator import attrgetter

def get_size(a):
    h, w = a.shape[:2]
    return w, h

    
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2


MIN_MATCH_COUNT = 10

class App:
    def __init__(self, src):
        self.cap = video.create_capture(src)
        self.ref_frame  = None

        self.detector = cv2.ORB( nfeatures = 1000 )
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)

        cv2.namedWindow('plane')
        self.rect_sel = common.RectSelector('plane', self.on_rect)

        self.frame = None

    def match_frames(self):
        if len(self.frame_desc) < MIN_MATCH_COUNT or len(self.frame_desc) < MIN_MATCH_COUNT:
            return
        
        raw_matches = self.matcher.knnMatch(self.ref_descs, trainDescriptors = self.frame_desc, k = 2)
        p0, p1 = [], []
        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                m = m[0]
                p0.append( self.ref_points[m.queryIdx].pt )
                p1.append( self.frame_points[m.trainIdx].pt )
        p0, p1 = np.float32((p0, p1))
        if len(p0) < MIN_MATCH_COUNT:
            return

        H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 4.0)
        status = status.ravel() != 0
        if status.sum() < MIN_MATCH_COUNT:
            return
        p0, p1 = p0[status], p1[status]
        return p0, p1, H


    def on_frame(self, frame):
        if self.frame is None or not self.rect_sel.dragging:
            self.frame = frame = np.fliplr(frame).copy()
            self.frame_points, self.frame_desc = self.detector.detectAndCompute(self.frame, None)
            if self.frame_desc is None:  # detectAndCompute returns descs=None if not keypoints found
                self.frame_desc = []
        else:
            self.ref_frame = None

        w, h = get_size(self.frame)
        vis = np.zeros((h, w*2, 3), np.uint8)
        vis[:h,:w] = self.frame
        self.rect_sel.draw(vis)
        for kp in self.frame_points:
            x, y = kp.pt
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 255))
        
        if self.ref_frame is not None:
            vis[:h,w:] = self.ref_frame
            x0, y0, x1, y1 = self.ref_rect
            cv2.rectangle(vis, (x0+w, y0), (x1+w, y1), (0, 255, 0), 2)

            for kp in self.ref_points:
                x, y = kp.pt
                cv2.circle(vis, (int(x+w), int(y)), 2, (0, 255, 255))


            match = self.match_frames()
            if match is not None:
                p0, p1, H = match
                for (x0, y0), (x1, y1) in zip(np.int32(p0), np.int32(p1)):
                    cv2.line(vis, (x0+w, y0), (x1, y1), (0, 255, 0))
                x0, y0, x1, y1 = self.ref_rect
                corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H) )
                cv2.polylines(vis, [corners], True, (255, 255, 255), 2)
        
        cv2.imshow('plane', vis)

    def on_rect(self, rect):
        x0, y0, x1, y1 = rect
        self.ref_frame = self.frame.copy()
        self.ref_rect = rect
        points, descs = [], []
        for kp, desc in zip(self.frame_points, self.frame_desc):
            x, y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                points.append(kp)
                descs.append(desc)
        self.ref_points, self.ref_descs = points, np.uint8(descs)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            self.on_frame(frame)
            ch = cv2.waitKey(1)
            if ch == 27:
                break

if __name__ == '__main__':
    print __doc__

    import sys
    try: video_src = sys.argv[1]
    except: video_src = '0'
    App(video_src).run()
