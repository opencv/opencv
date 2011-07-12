#!/usr/bin/python
import urllib2
import sys
import time
from math import cos, sin
import cv2.cv as cv

CLOCKS_PER_SEC = 1.0
MHI_DURATION = 1
MAX_TIME_DELTA = 0.5
MIN_TIME_DELTA = 0.05
N = 4
buf = range(10) 
last = 0
mhi = None # MHI
orient = None # orientation
mask = None # valid orientation mask
segmask = None # motion segmentation map
storage = None # temporary storage

def update_mhi(img, dst, diff_threshold):
    global last
    global mhi
    global storage
    global mask
    global orient
    global segmask
    timestamp = time.clock() / CLOCKS_PER_SEC # get current time in seconds
    size = cv.GetSize(img) # get current frame size
    idx1 = last
    if not mhi or cv.GetSize(mhi) != size:
        for i in range(N):
            buf[i] = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
            cv.Zero(buf[i])
        mhi = cv.CreateImage(size,cv. IPL_DEPTH_32F, 1)
        cv.Zero(mhi) # clear MHI at the beginning
        orient = cv.CreateImage(size,cv. IPL_DEPTH_32F, 1)
        segmask = cv.CreateImage(size,cv. IPL_DEPTH_32F, 1)
        mask = cv.CreateImage(size,cv. IPL_DEPTH_8U, 1)
    
    cv.CvtColor(img, buf[last], cv.CV_BGR2GRAY) # convert frame to grayscale
    idx2 = (last + 1) % N # index of (last - (N-1))th frame
    last = idx2
    silh = buf[idx2]
    cv.AbsDiff(buf[idx1], buf[idx2], silh) # get difference between frames
    cv.Threshold(silh, silh, diff_threshold, 1, cv.CV_THRESH_BINARY) # and threshold it
    cv.UpdateMotionHistory(silh, mhi, timestamp, MHI_DURATION) # update MHI
    cv.CvtScale(mhi, mask, 255./MHI_DURATION,
                (MHI_DURATION - timestamp)*255./MHI_DURATION)
    cv.Zero(dst)
    cv.Merge(mask, None, None, None, dst)
    cv.CalcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3)
    if not storage:
        storage = cv.CreateMemStorage(0)
    seq = cv.SegmentMotion(mhi, segmask, storage, timestamp, MAX_TIME_DELTA)
    for (area, value, comp_rect) in seq:
        if comp_rect[2] + comp_rect[3] > 100: # reject very small components
            color = cv.CV_RGB(255, 0,0)
            silh_roi = cv.GetSubRect(silh, comp_rect)
            mhi_roi = cv.GetSubRect(mhi, comp_rect)
            orient_roi = cv.GetSubRect(orient, comp_rect)
            mask_roi = cv.GetSubRect(mask, comp_rect)
            angle = 360 - cv.CalcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION)

            count = cv.Norm(silh_roi, None, cv.CV_L1, None) # calculate number of points within silhouette ROI
            if count < (comp_rect[2] * comp_rect[3] * 0.05):
                continue

            magnitude = 30.
            center = ((comp_rect[0] + comp_rect[2] / 2), (comp_rect[1] + comp_rect[3] / 2))
            cv.Circle(dst, center, cv.Round(magnitude*1.2), color, 3, cv.CV_AA, 0)
            cv.Line(dst,
                    center,
                    (cv.Round(center[0] + magnitude * cos(angle * cv.CV_PI / 180)),
                     cv.Round(center[1] - magnitude * sin(angle * cv.CV_PI / 180))),
                    color,
                    3,
                    cv.CV_AA,
                    0)

if __name__ == "__main__":
    motion = 0
    capture = 0

    if len(sys.argv)==1:
        capture = cv.CreateCameraCapture(0)
    elif len(sys.argv)==2 and sys.argv[1].isdigit():
        capture = cv.CreateCameraCapture(int(sys.argv[1]))
    elif len(sys.argv)==2:
        capture = cv.CreateFileCapture(sys.argv[1]) 

    if not capture:
        print "Could not initialize capturing..."
        sys.exit(-1)
        
    cv.NamedWindow("Motion", 1)
    while True:
        image = cv.QueryFrame(capture)
        if(image):
            if(not motion):
                    motion = cv.CreateImage((image.width, image.height), 8, 3)
                    cv.Zero(motion)
                    #motion.origin = image.origin
            update_mhi(image, motion, 30)
            cv.ShowImage("Motion", motion)
            if(cv.WaitKey(10) != -1):
                break
        else:
            break
    cv.DestroyWindow("Motion")
