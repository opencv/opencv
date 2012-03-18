#!/usr/bin/python
import sys
import urllib2
import cv2.cv as cv

src=None
dst=None
src2=None

def on_mouse(event, x, y, flags, param):

    if not src:
        return

    if event==cv.CV_EVENT_LBUTTONDOWN:
        cv.LogPolar(src, dst, (x, y), 40, cv.CV_INTER_LINEAR + cv.CV_WARP_FILL_OUTLIERS)
        cv.LogPolar(dst, src2, (x, y), 40, cv.CV_INTER_LINEAR + cv.CV_WARP_FILL_OUTLIERS + cv.CV_WARP_INVERSE_MAP)
        cv.ShowImage("log-polar", dst)
        cv.ShowImage("inverse log-polar", src2)

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        src = cv.LoadImage( sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
    else:
        url = 'http://code.opencv.org/svn/opencv/trunk/opencv/samples/c/fruits.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        src = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_COLOR)
        
    cv.NamedWindow("original", 1)
    cv.NamedWindow("log-polar", 1)
    cv.NamedWindow("inverse log-polar", 1)
  
    
    dst = cv.CreateImage((256, 256), 8, 3)
    src2 = cv.CreateImage(cv.GetSize(src), 8, 3)
    
    cv.SetMouseCallback("original", on_mouse)
    on_mouse(cv.CV_EVENT_LBUTTONDOWN, src.width/2, src.height/2, None, None)
    
    cv.ShowImage("original", src)
    cv.WaitKey()
    cv.DestroyAllWindows()
