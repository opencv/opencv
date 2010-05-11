#!/usr/bin/python
import sys
from opencv.cv import *
from opencv.highgui import *

wndname = "Distance transform";
tbarname = "Threshold";

# The output images
dist = 0;
dist8u1 = 0;
dist8u2 = 0;
dist8u = 0;
dist32s = 0;

gray = 0;
edge = 0;

# define a trackbar callback
def on_trackbar( edge_thresh ):

    cvThreshold( gray, edge, float(edge_thresh), float(edge_thresh), CV_THRESH_BINARY );
    #Distance transform                  
    cvDistTransform( edge, dist, CV_DIST_L2, CV_DIST_MASK_5, None, None );

    cvConvertScale( dist, dist, 5000.0, 0 );
    cvPow( dist, dist, 0.5 );
    
    cvConvertScale( dist, dist32s, 1.0, 0.5 );
    cvAndS( dist32s, cvScalarAll(255), dist32s, None );
    cvConvertScale( dist32s, dist8u1, 1, 0 );
    cvConvertScale( dist32s, dist32s, -1, 0 );
    cvAddS( dist32s, cvScalarAll(255), dist32s, None );
    cvConvertScale( dist32s, dist8u2, 1, 0 );
    cvMerge( dist8u1, dist8u2, dist8u2, None, dist8u );
    cvShowImage( wndname, dist8u );


if __name__ == "__main__":
    edge_thresh = 100;

    filename = "../c/stuff.jpg"
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    gray = cvLoadImage( filename, 0 )
    if not gray:
        print "Failed to load %s" % filename
        sys.exit(-1)

    # Create the output image
    dist = cvCreateImage( cvSize(gray.width,gray.height), IPL_DEPTH_32F, 1 );
    dist8u1 = cvCloneImage( gray );
    dist8u2 = cvCloneImage( gray );
    dist8u = cvCreateImage( cvSize(gray.width,gray.height), IPL_DEPTH_8U, 3 );
    dist32s = cvCreateImage( cvSize(gray.width,gray.height), IPL_DEPTH_32S, 1 );

    # Convert to grayscale
    edge = cvCloneImage( gray );

    # Create a window
    cvNamedWindow( wndname, 1 );

    # create a toolbar 
    cvCreateTrackbar( tbarname, wndname, edge_thresh, 255, on_trackbar );

    # Show the image
    on_trackbar(edge_thresh);

    # Wait for a key stroke; the same function arranges events processing
    cvWaitKey(0);
