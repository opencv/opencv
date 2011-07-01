import numpy as np
import cv2, cv

help_message = '''
  USAGE: watershed.py [<image>]

  Use keys 1 - 7 to switch marker color
  SPACE - update segmentation
  r     - reset
  a     - switch autoupdate
  ESC   - exit

'''

class App:
    def __init__(self, fn):
        self.img = cv2.imread(fn)
        h, w = self.img.shape[:2]
        self.markers = np.zeros((h, w), np.int32)
        self.markers_vis = self.img.copy()
        self.cur_marker = 1
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255

        cv2.imshow('img', self.markers_vis)

        self.prev_pt = None
        self.need_update = False
        self.auto_update = True
        cv2.setMouseCallback('img', self.onmouse)


    def onmouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.CV_EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        if self.prev_pt and flags & cv.CV_EVENT_FLAG_LBUTTON:
            color = map(int, self.colors[self.cur_marker])
            cv.Line(self.markers, self.prev_pt, pt, self.cur_marker, 5)
            cv.Line(self.markers_vis, self.prev_pt, pt, color, 5)
            self.need_update = True
            self.prev_pt = pt
            cv2.imshow('img', self.markers_vis)
        else:
            self.prev_pt = None

    def watershed(self):
        m = self.markers.copy()
        cv2.watershed(self.img, m)
        vis = np.uint8( (self.img + self.colors[np.maximum(m, 0)]) / 2 )
        cv2.imshow('watershed', vis)
        self.need_update = False

    def run(self):
        while True:
            ch = cv2.waitKey(10)
            if ch == 27:
                break
            if ch >= ord('1') and ch <= ord('7'):
                self.cur_marker = ch - ord('0')
                print 'marker: ', self.cur_marker
            if ch == ord(' ') or (self.need_update and self.auto_update):
                self.watershed()
            if ch in [ord('a'), ord('A')]:
                self.auto_update = not self.auto_update
                print 'auto_update if', ['off', 'on'][self.auto_update]
            if ch in [ord('r'), ord('R')]:
                self.markers = np.zeros(self.img.shape[:2], np.int32)
                self.markers_vis = self.img.copy()
                cv2.imshow('img', self.markers_vis)
                cv2.destroyWindow('watershed')


if __name__ == '__main__':
    import sys
    try: fn = sys.argv[1]
    except: fn = '../cpp/fruits.jpg'
    print help_message
    App(fn).run()
