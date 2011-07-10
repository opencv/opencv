import numpy as np
import cv2, cv
import video
from common import anorm2, draw_str
from time import clock

help_message = '''
USAGE: lk_track.py [<video_source>]

Keys:
  1 - toggle old/new CalcOpticalFlowPyrLK implementation
  SPACE - reset features
'''




lk_params = dict( winSize  = (3, 3), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                  derivLambda = 0.0 )    

feature_params = dict( maxCorners = 1000, 
                       qualityLevel = 0.1,
                       minDistance = 5,
                       blockSize = 5 )

def calc_flow_old(img0, img1, p0):
    p0 = [(x, y) for x, y in p0.reshape(-1, 2)]
    h, w = img0.shape[:2]
    img0_cv = cv.CreateMat(h, w, cv.CV_8U)
    img1_cv = cv.CreateMat(h, w, cv.CV_8U)
    np.asarray(img0_cv)[:] = img0
    np.asarray(img1_cv)[:] = img1
    t = clock()
    features, status, error  = cv.CalcOpticalFlowPyrLK(img0_cv, img1_cv, None, None, p0, 
        lk_params['winSize'], lk_params['maxLevel'], (cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 0.03), 0, p0)
    return np.float32(features), status, error, clock()-t

def main():
    import sys
    try: video_src = sys.argv[1]
    except: video_src = video.presets['chess']

    cv2.namedWindow('img', 0)

    track_len = 4
    tracks = []
    cam = video.create_capture(video_src)
    old_mode = True
    while True:
        ret, frame = cam.read()
        vis = frame.copy()
        if len(tracks) > 0:
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            img0 = cv2.cvtColor(prev_frame, cv.CV_BGR2GRAY)
            img1 = cv2.cvtColor(frame, cv.CV_BGR2GRAY)
            if old_mode:
                p1,  st, err, dt = calc_flow_old(img0, img1, p0)
            else:
                t = clock()
                p1,  st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, **lk_params)
                dt = clock()-t
            for tr, (x, y) in zip(tracks, p1.reshape(-1, 2)):
                tr.append((x, y))
                if len(tr) > 10:
                    del tr[0]
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
            draw_str(vis, (20, 20), ['new', 'old'][old_mode]+' mode')
            draw_str(vis, (20, 40), 'time: %.02f ms' % (dt*1000))
        prev_frame = frame.copy()

        cv2.imshow('img', vis)
        ch = cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord(' ') or len(tracks) == 0:
            gray = cv2.cvtColor(frame, cv.CV_BGR2GRAY)
            p = cv2.goodFeaturesToTrack(gray, **feature_params)
            p = [] if p is None else p.reshape(-1, 2)
            tracks = []
            for x, y in np.float32(p):
                tracks.append([(x, y)])
        if ch == ord('1'):
            old_mode = not old_mode

if __name__ == '__main__':
    main()
