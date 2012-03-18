#!/usr/bin/python
import urllib2
import sys
import cv2.cv as cv

class Sketcher:
    def __init__(self, windowname, dests):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        cv.SetMouseCallback(self.windowname, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.CV_EVENT_LBUTTONUP or not (flags & cv.CV_EVENT_FLAG_LBUTTON):
            self.prev_pt = None
        elif event == cv.CV_EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.CV_EVENT_MOUSEMOVE and (flags & cv.CV_EVENT_FLAG_LBUTTON) :
            if self.prev_pt:
                for dst in self.dests:
                    cv.Line(dst, self.prev_pt, pt, cv.ScalarAll(255), 5, 8, 0)
            self.prev_pt = pt
            cv.ShowImage(self.windowname, img)

if __name__=="__main__":
    if len(sys.argv) > 1:
        img0 = cv.LoadImage( sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
    else:
        url = 'http://code.opencv.org/svn/opencv/trunk/opencv/samples/c/fruits.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        img0 = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_COLOR)

    print "Hot keys:"
    print "\tESC - quit the program"
    print "\tr - restore the original image"
    print "\ti or ENTER - run inpainting algorithm"
    print "\t\t(before running it, paint something on the image)"
    
    cv.NamedWindow("image", 1)
    cv.NamedWindow("inpainted image", 1)

    img = cv.CloneImage(img0)
    inpainted = cv.CloneImage(img0)
    inpaint_mask = cv.CreateImage(cv.GetSize(img), 8, 1)

    cv.Zero(inpaint_mask)
    cv.Zero(inpainted)
    cv.ShowImage("image", img)
    cv.ShowImage("inpainted image", inpainted)

    sk = Sketcher("image", [img, inpaint_mask])
    while True:
        c = cv.WaitKey(0) % 0x100

        if c == 27 or c == ord('q'):
            break

        if c == ord('r'):
            cv.Zero(inpaint_mask)
            cv.Copy(img0, img)
            cv.ShowImage("image", img)

        if c == ord('i') or c == ord('\n'):
            cv.Inpaint(img, inpaint_mask, inpainted, 3, cv.CV_INPAINT_TELEA)
            cv.ShowImage("inpainted image", inpainted)
    cv.DestroyAllWindows()
