import numpy as np
import cv2, cv
import video
import sys

try: fn = sys.argv[1]
except: fn = video.presets['chess']

cam = video.create_capture(fn)
ret, prev = cam.read()
#prev = cv2.pyrDown(prev)
prevgray = cv2.cvtColor(prev, cv.CV_BGR2GRAY)

def draw_flow(img, flow, step):
    h, w = img.shape[:2]
    y, x = map(np.ravel, np.mgrid[step/2:h:step, step/2:w:step])
    f = flow[y,x]
    x1 = x + f[:,0]
    y1 = y + f[:,1]
    #lines = np.int32( np.vstack([x, y, x1, y1]).T )
    vis = cv2.cvtColor(img, cv.CV_GRAY2BGR)
    #print lines
    #cv2.polylines(vis, lines, 0, (0, 255, 0))
    for x_, y_, x1_, y1_ in np.int32(zip(x, y, x1, y1)):
        cv2.line(vis, (x_, y_), (x1_, y1_), (0, 255, 0))
    return vis

while True:
    ret, img = cam.read()
    #img = cv2.pyrDown(img)
    gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray

    cv2.imshow('flow', draw_flow(gray, flow, 16))
    if cv2.waitKey(5) == 27:
        break

