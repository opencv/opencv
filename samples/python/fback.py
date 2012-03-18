#!/usr/bin/env python

from cv import *

class FBackDemo:
    def __init__(self):
        self.capture = CaptureFromCAM(0)
        self.mv_step = 16
        self.mv_scale = 1.5
        self.mv_color = (0, 255, 0)
        self.cflow = None
        self.flow = None
        
        NamedWindow( "Optical Flow", 1 )

        print( "Press ESC - quit the program\n" )

    def draw_flow(self, flow, prevgray):
        """ Returns a nice representation of a hue histogram """

        CvtColor(prevgray, self.cflow, CV_GRAY2BGR)
        for y in range(0, flow.height, self.mv_step):
            for x in range(0, flow.width, self.mv_step):
                fx, fy = flow[y, x]
                Line(self.cflow, (x,y), (x+fx,y+fy), self.mv_color)
                Circle(self.cflow, (x,y), 2, self.mv_color, -1)
        ShowImage("Optical Flow", self.cflow)

    def run(self):
        first_frame = True
        
        while True:
            frame = QueryFrame( self.capture )

            if first_frame:
                gray = CreateImage(GetSize(frame), 8, 1)
                prev_gray = CreateImage(GetSize(frame), 8, 1)
                flow = CreateImage(GetSize(frame), 32, 2)
                self.cflow = CreateImage(GetSize(frame), 8, 3)
                
            CvtColor(frame, gray, CV_BGR2GRAY)
            if not first_frame:
                CalcOpticalFlowFarneback(prev_gray, gray, flow,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                self.draw_flow(flow, prev_gray)
                c = WaitKey(7)
                if c in [27, ord('q'), ord('Q')]:
                    break
            prev_gray, gray = gray, prev_gray        
            first_frame = False

if __name__=="__main__":
    demo = FBackDemo()
    demo.run()
    cv.DestroyAllWindows()
