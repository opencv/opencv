#!/usr/bin/python
from opencv.cv import *
from opencv.highgui import *
import sys

inpaint_mask = None
img0 = None
img = None
inpainted = None
prev_pt = cvPoint(-1,-1)

def on_mouse( event, x, y, flags, param ):
    global prev_pt
    if not img:
        return

    if event == CV_EVENT_LBUTTONUP or not (flags & CV_EVENT_FLAG_LBUTTON):
        prev_pt = cvPoint(-1,-1)
    elif event == CV_EVENT_LBUTTONDOWN:
        prev_pt = cvPoint(x,y)
    elif event == CV_EVENT_MOUSEMOVE and (flags & CV_EVENT_FLAG_LBUTTON) :
        pt = cvPoint(x,y)
        if prev_pt.x < 0:
            prev_pt = pt
        cvLine( inpaint_mask, prev_pt, pt, cvScalarAll(255), 5, 8, 0 )
        cvLine( img, prev_pt, pt, cvScalarAll(255), 5, 8, 0 )
        prev_pt = pt
        cvShowImage( "image", img )

if __name__=="__main__":
    filename = "../c/fruits.jpg"
    if len(sys.argv) >= 2:
    	filename = sys.argv[1]

    img0 = cvLoadImage(filename,-1)
    if not img0:
    	print "Can't open image '%s'" % filename
        sys.exit(1)

    print "Hot keys:"
    print "\tESC - quit the program"
    print "\tr - restore the original image"
    print "\ti or ENTER - run inpainting algorithm"
    print "\t\t(before running it, paint something on the image)"
    
    cvNamedWindow( "image", 1 )

    img = cvCloneImage( img0 )
    inpainted = cvCloneImage( img0 )
    inpaint_mask = cvCreateImage( cvGetSize(img), 8, 1 )

    cvZero( inpaint_mask )
    cvZero( inpainted )
    cvShowImage( "image", img )
    cvShowImage( "watershed transform", inpainted )
    cvSetMouseCallback( "image", on_mouse, None )

    while True:
        c = cvWaitKey(0)

        if c == '\x1b' or c == 'q':
            break

        if c == 'r':
            cvZero( inpaint_mask )
            cvCopy( img0, img )
            cvShowImage( "image", img )

        if c == 'i' or c == '\012':
            cvNamedWindow( "inpainted image", 1 )
            cvInpaint( img, inpaint_mask, inpainted, 3, CV_INPAINT_TELEA )
            cvShowImage( "inpainted image", inpainted )

