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
from collections import namedtuple
from common import getsize

    
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10


ar_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                       [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], 
                       [0.5, 0.5, 2]])
ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0), 
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7), 
            (4, 8), (5, 8), (6, 8), (7, 8)]



def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
            x, y = kp.pt
            cv2.circle(vis, (int(x), int(y)), 2, color)

class App:
    def __init__(self, src):
        self.cap = video.create_capture(src)
        self.frame = None
        self.paused = False
        self.ref_frame  = None

        self.detector = cv2.ORB( nfeatures = 1000 )
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)

        cv2.namedWindow('plane')
        self.rect_sel = common.RectSelector('plane', self.on_rect)


    def match_frames(self):
        if len(self.frame_desc) < MIN_MATCH_COUNT or len(self.frame_desc) < MIN_MATCH_COUNT:
            return
        
        raw_matches = self.matcher.knnMatch(self.frame_desc, k = 2)
        p0, p1 = [], []
        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                m = m[0]
                p0.append( self.ref_points[m.trainIdx].pt )  # queryIdx
                p1.append( self.frame_points[m.queryIdx].pt )
        p0, p1 = np.float32((p0, p1))
        if len(p0) < MIN_MATCH_COUNT:
            return

        H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 4.0)
        status = status.ravel() != 0
        if status.sum() < MIN_MATCH_COUNT:
            return
        p0, p1 = p0[status], p1[status]
        return p0, p1, H


    def on_frame(self, vis):
        match = self.match_frames()
        if match is None:
            return
        w, h = getsize(self.frame)
        p0, p1, H = match
        for (x0, y0), (x1, y1) in zip(np.int32(p0), np.int32(p1)):
            cv2.line(vis, (x0+w, y0), (x1, y1), (0, 255, 0))
        x0, y0, x1, y1 = self.ref_rect
        corners0 = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        img_corners = cv2.perspectiveTransform(corners0.reshape(1, -1, 2), H)
        cv2.polylines(vis, [np.int32(img_corners)], True, (255, 255, 255), 2)

        corners3d = np.hstack([corners0, np.zeros((4, 1), np.float32)])
        fx = 0.9
        K = np.float64([[fx*w, 0, 0.5*(w-1)],
                        [0, fx*w, 0.5*(h-1)],
                        [0.0,0.0,      1.0]])
        dist_coef = np.zeros(4)
        ret, rvec, tvec = cv2.solvePnP(corners3d, img_corners, K, dist_coef)
        verts = ar_verts * [(x1-x0), (y1-y0), -(x1-x0)*0.3] + (x0, y0, 0)
        verts = cv2.projectPoints(verts, rvec, tvec, K, dist_coef)[0].reshape(-1, 2)
        for i, j in ar_edges:
            (x0, y0), (x1, y1) = verts[i], verts[j]
            cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 2)

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

        self.matcher.clear()
        self.matcher.add([self.ref_descs])

    def run(self):
        while True:
            playing = not self.paused and not self.rect_sel.dragging
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = np.fliplr(frame).copy()
                self.frame_points, self.frame_desc = self.detector.detectAndCompute(self.frame, None)
                if self.frame_desc is None:  # detectAndCompute returns descs=None if not keypoints found
                    self.frame_desc = []
            
            w, h = getsize(self.frame)
            vis = np.zeros((h, w*2, 3), np.uint8)
            vis[:h,:w] = self.frame
            if self.ref_frame is not None:
                vis[:h,w:] = self.ref_frame
                x0, y0, x1, y1 = self.ref_rect
                cv2.rectangle(vis, (x0+w, y0), (x1+w, y1), (0, 255, 0), 2)
                draw_keypoints(vis[:,w:], self.ref_points)
            draw_keypoints(vis, self.frame_points)

            if playing and self.ref_frame is not None:
                self.on_frame(vis)
            
            self.rect_sel.draw(vis)
            cv2.imshow('plane', vis)
            ch = cv2.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == 27:
                break

if __name__ == '__main__':
    print __doc__

    import sys
    try: video_src = sys.argv[1]
    except: video_src = 0
    App(video_src).run()
