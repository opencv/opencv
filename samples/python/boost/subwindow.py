#!/usr/bin/env python

import opencv

class SubwindowDemo:

    def __init__(self):
        self.capture = opencv.VideoCapture(0)
        self.capture.set(opencv.CV_CAP_PROP_FRAME_HEIGHT, 1000)
        self.capture.set(opencv.CV_CAP_PROP_FRAME_WIDTH, 1000)
        opencv.namedWindow( "Video", opencv.CV_WINDOW_KEEPRATIO)
        opencv.namedWindow( "Track Window", opencv.CV_WINDOW_KEEPRATIO)

        opencv.setMouseCallback( "Video", self.on_mouse,None)

        self.drag_start = None      # Set to (x,y) when mouse starts drag
        self.track_window = None    # Set to rect when the mouse drag finishes

        print( "Keys:\n"
            "    ESC,q - quit the program\n"
            "To initialize the subwindow, drag across the image with the mouse\n" )
    def __del__(self):
        opencv.setMouseCallback( "Video", None,None)

    def on_mouse(self, event, x, y, flags, param):
        #print "caught mouse", event,x,y,flags,param
        if event == opencv.CV_EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if event == opencv.CV_EVENT_LBUTTONUP:
            self.drag_start = None
            if 0 not in self.selection:
                self.track_window = self.selection
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax - xmin, ymax - ymin)

    def run(self):
        img = opencv.Mat()
        img_sub = opencv.Mat()
        while True:
            #grab a frame
            self.capture.read(img)
            #uses imshow
            opencv.imshow("Video",img)
            
            if self.track_window:
                #show a sub region
                img_sub = img.roi(opencv.Rect(*self.track_window))
                opencv.imshow("Track Window",img_sub)
            
            #wait for a key, returns an int
            key = opencv.waitKey(10)
            if key in ( 27, ord('q')):
                break
            


if __name__=="__main__":
    demo = SubwindowDemo()
    demo.run()
    opencv.setMouseCallback( "Video", None,None)
