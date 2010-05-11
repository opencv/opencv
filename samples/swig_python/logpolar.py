#!/usr/bin/python
import sys
from opencv.cv import *
from opencv.highgui import *

src=None
dst=None
src2=None

def on_mouse( event, x, y, flags, param ):

    if( not src ):
        return;

    if event==CV_EVENT_LBUTTONDOWN:
        cvLogPolar( src, dst, cvPoint2D32f(x,y), 40, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS );
        cvLogPolar( dst, src2, cvPoint2D32f(x,y), 40, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP );
        cvShowImage( "log-polar", dst );
        cvShowImage( "inverse log-polar", src2 );

if __name__ == "__main__":
    
    filename = "../c/fruits.jpg"
    if len(sys.argv)>1:
        filename=argv[1]
    
    src = cvLoadImage(filename,1)
    if not src:
        print "Could not open %s" % filename
        sys.exit(-1)
        
    cvNamedWindow( "original",1 );
    cvNamedWindow( "log-polar", 1 );
    cvNamedWindow( "inverse log-polar", 1 );
  
    
    dst = cvCreateImage( cvSize(256,256), 8, 3 );
    src2 = cvCreateImage( cvGetSize(src), 8, 3 );
    
    cvSetMouseCallback( "original", on_mouse );
    on_mouse( CV_EVENT_LBUTTONDOWN, src.width/2, src.height/2, None, None)
    
    cvShowImage( "original", src );
    cvWaitKey();
