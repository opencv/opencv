import numpy as np
import cv2, cv
import video
from common import nothing, clock, draw_str

MHI_DURATION = 1.0
DEFAULT_THRESHOLD = 16
MAX_TIME_DELTA = 0.5
MIN_TIME_DELTA = 0.05


if __name__ == '__main__':
    import sys
    try: video_src = sys.argv[1]
    except: video_src = 'synth:class=chess:bg=../cpp/lena.jpg:noise=0.01'

    cv2.namedWindow('motempl')
    visuals = ['input', 'frame_diff', 'motion_hist', 'grad_orient']
    cv2.createTrackbar('visual', 'motempl', 2, len(visuals)-1, nothing)
    cv2.createTrackbar('threshold', 'motempl', DEFAULT_THRESHOLD, 255, nothing)

    cam = video.create_capture(video_src)
    ret, frame = cam.read()
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:,:,1] = 255
    while True:
        ret, frame = cam.read()
        frame_diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv.CV_BGR2GRAY)
        thrs = cv2.getTrackbarPos('threshold', 'motempl')
        ret, motion_mask = cv2.threshold(gray_diff, thrs, 255, cv2.THRESH_BINARY)
        timestamp = clock()
        cv2.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
        mg_mask, mg_orient = cv2.calcMotionGradient( motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=5 );

        visual_name = visuals[cv2.getTrackbarPos('visual', 'motempl')]
        if visual_name == 'input':
            vis = frame.copy()
        elif visual_name == 'frame_diff':
            vis = frame_diff.copy()
        elif visual_name == 'motion_hist':
            vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
            vis = cv2.cvtColor(vis, cv.CV_GRAY2BGR)
        elif visual_name == 'grad_orient':
            hsv[:,:,0] = mg_orient/2
            hsv[:,:,2] = mg_mask*255
            vis = cv2.cvtColor(hsv, cv.CV_HSV2BGR)
        draw_str(vis, (20, 20), visual_name)
        cv2.imshow('motempl', vis)

        prev_frame = frame.copy()
        if cv2.waitKey(5) == 27:
            break