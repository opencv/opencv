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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img0 = cv.LoadImage( sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
    else:
        url = 'https://raw.github.com/Itseez/opencv/master/samples/c/fruits.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        img0 = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_COLOR)

    rng = cv.RNG(-1)

    print "Hot keys:"
    print "\tESC - quit the program"
    print "\tr - restore the original image"
    print "\tw - run watershed algorithm"
    print "\t  (before that, roughly outline several markers on the image)"

    cv.NamedWindow("image", 1)
    cv.NamedWindow("watershed transform", 1)

    img = cv.CloneImage(img0)
    img_gray = cv.CloneImage(img0)
    wshed = cv.CloneImage(img0)
    marker_mask = cv.CreateImage(cv.GetSize(img), 8, 1)
    markers = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32S, 1)

    cv.CvtColor(img, marker_mask, cv.CV_BGR2GRAY)
    cv.CvtColor(marker_mask, img_gray, cv.CV_GRAY2BGR)

    cv.Zero(marker_mask)
    cv.Zero(wshed)

    cv.ShowImage("image", img)
    cv.ShowImage("watershed transform", wshed)

    sk = Sketcher("image", [img, marker_mask])

    while True:
        c = cv.WaitKey(0) % 0x100
        if c == 27 or c == ord('q'):
            break
        if c == ord('r'):
            cv.Zero(marker_mask)
            cv.Copy(img0, img)
            cv.ShowImage("image", img)
        if c == ord('w'):
            storage = cv.CreateMemStorage(0)
            #cv.SaveImage("wshed_mask.png", marker_mask)
            #marker_mask = cv.LoadImage("wshed_mask.png", 0)
            contours = cv.FindContours(marker_mask, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
            def contour_iterator(contour):
                while contour:
                    yield contour
                    contour = contour.h_next()

            cv.Zero(markers)
            comp_count = 0
            for c in contour_iterator(contours):
                cv.DrawContours(markers,
                                c,
                                cv.ScalarAll(comp_count + 1),
                                cv.ScalarAll(comp_count + 1),
                                -1,
                                -1,
                                8)
                comp_count += 1

            cv.Watershed(img0, markers)

            cv.Set(wshed, cv.ScalarAll(255))

            # paint the watershed image
            color_tab = [(cv.RandInt(rng) % 180 + 50, cv.RandInt(rng) % 180 + 50, cv.RandInt(rng) % 180 + 50) for i in range(comp_count)]
            for j in range(markers.height):
                for i in range(markers.width):
                    idx = markers[j, i]
                    if idx != -1:
                        wshed[j, i] = color_tab[int(idx - 1)]

            cv.AddWeighted(wshed, 0.5, img_gray, 0.5, 0, wshed)
            cv.ShowImage("watershed transform", wshed)
    cv.DestroyAllWindows()
